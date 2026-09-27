import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn


COMFY_ROOT = Path(__file__).resolve().parents[3]
CUSTOM_NODES_ROOT = COMFY_ROOT / "custom_nodes"
PACKAGE_NAME = Path(__file__).resolve().parents[1].name
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(CUSTOM_NODES_ROOT))

import comfy.model_patcher
from comfy.weight_adapter.lora import LoRAAdapter

lora_nodes = importlib.import_module(f"{PACKAGE_NAME}.int8_lora")
lora_patching = importlib.import_module(f"{PACKAGE_NAME}.int8_lora_patching")
model_adapter = importlib.import_module(f"{PACKAGE_NAME}.int8_model_adapter")
quant = importlib.import_module(f"{PACKAGE_NAME}.int8_quant")


WEIGHT_KEY = "diffusion_model.layer.weight"
RANK = 2


class PooledStackDiffusionModel(torch.nn.Module):
	def __init__(self, diffusion_model):
		super().__init__()
		self.diffusion_model = diffusion_model

	def forward(self, x):
		return self.diffusion_model.layer(x)


def _build_int8_patcher():
	layer = quant.Int8TensorwiseOps.Linear(8, 4, bias=False, dtype=torch.float32)
	layer.weight = nn.Parameter((torch.arange(32, dtype=torch.int8).reshape(4, 8) % 9) - 4, requires_grad=False)
	# 0.04 turns the fixed 0.1 LoRA factors below into a half-step delta, so the
	# rounding each Apply LoRA Stack performs is visible in the baked weight.
	layer.weight_scale = torch.tensor(0.04)
	layer._is_quantized = True
	layer._quant_format = "int8_tensorwise"
	diffusion_model = nn.Module()
	diffusion_model.layer = layer
	return comfy.model_patcher.ModelPatcher(
		PooledStackDiffusionModel(diffusion_model),
		torch.device("cpu"),
		torch.device("cpu"),
	)


def _lora_adapter():
	up = torch.full((4, RANK), 0.1)
	down = torch.full((RANK, 8), 0.1)
	return LoRAAdapter([], (up, down, None, None, None, None))


def _apply_stack(patcher, loras, pool_stochastic_stacks):
	adapters = {name: _lora_adapter() for name, _strength in loras}

	def load_torch_file(path, safe_load=True):
		return path

	def load_lora(data, key_map, log_missing=True):
		return {WEIGHT_KEY: adapters[data]}

	with mock.patch.object(lora_nodes, "_get_key_map", return_value={}):
		with mock.patch.object(lora_nodes.folder_paths, "get_full_path", side_effect=lambda folder, name: name):
			with mock.patch.object(lora_nodes.comfy.utils, "load_torch_file", side_effect=load_torch_file):
				with mock.patch.object(lora_nodes.comfy.lora, "load_lora", side_effect=load_lora):
					return lora_nodes.INT8LoraLoaderStack().apply_loras(
						lora_nodes.LORA_MODE_STOCHASTIC,
						patcher,
						list(loras),
						pool_stochastic_stacks=pool_stochastic_stacks,
					)[0]


def _materialize_weight(patcher):
	# The pooling contract is about how many stochastic rounding steps the layer sees
	# between the last Apply LoRA Stack and weight materialization.
	with mock.patch.object(quant, "_apply_int8_delta_inplace", wraps=quant._apply_int8_delta_inplace) as rounds:
		patcher.patch_weight_to_device(WEIGHT_KEY)
	return rounds.call_count, patcher.model.diffusion_model.layer.weight.detach().clone()


class PooledLoraStackTests(unittest.TestCase):
	def test_pooled_stacks_requantize_once_and_match_one_combined_stack(self):
		style = [("style.safetensors", 1.0)]
		inpaint = [("inpaint.safetensors", 1.0)]

		independent = _build_int8_patcher()
		independent = _apply_stack(_apply_stack(independent, style, False), inpaint, False)
		independent_rounds, independent_weight = _materialize_weight(independent)

		pooled = _build_int8_patcher()
		pooled = _apply_stack(_apply_stack(pooled, style, True), inpaint, True)
		pooled_rounds, pooled_weight = _materialize_weight(pooled)

		combined = _build_int8_patcher()
		combined = _apply_stack(combined, style + inpaint, True)
		combined_rounds, combined_weight = _materialize_weight(combined)

		self.assertEqual(independent_rounds, 2)
		self.assertEqual(pooled_rounds, 1)
		self.assertEqual(combined_rounds, 1)
		self.assertTrue(torch.equal(pooled_weight, combined_weight))
		self.assertFalse(torch.equal(pooled_weight, independent_weight))

	def test_pooled_merge_keeps_one_layer_entry_and_leaves_the_source_branch_alone(self):
		style_model = _build_int8_patcher()
		style_pooled = _apply_stack(style_model, [("style.safetensors", 0.5)], True)
		style_adapter = style_pooled.patches[WEIGHT_KEY][0][1]

		pooled = _apply_stack(style_pooled, [("inpaint.safetensors", 1.0)], True)

		self.assertEqual(len(pooled.patches[WEIGHT_KEY]), 1)
		self.assertEqual(len(style_pooled.patches[WEIGHT_KEY]), 1)
		self.assertIs(style_pooled.patches[WEIGHT_KEY][0][1], style_adapter)
		self.assertIsNot(pooled.patches[WEIGHT_KEY][0][1], style_adapter)
		self.assertEqual(style_model.patches, {})

	def test_pooling_is_ignored_outside_stochastic_mode(self):
		patcher = _build_int8_patcher()
		with self.assertLogs(level="INFO") as logs:
			with mock.patch.object(lora_nodes, "_dispatch_dynamic_stack", return_value=("dynamic",)) as dispatch:
				result = lora_nodes.INT8LoraLoaderStack().apply_loras(
					lora_nodes.LORA_MODE_DYNAMIC,
					patcher,
					[("style.safetensors", 0.5)],
					pool_stochastic_stacks=True,
				)

		dispatch.assert_called_once()
		self.assertEqual(result, ("dynamic",))
		self.assertIn("pooling only applies to Stochastic mode", "\n".join(logs.output))

	def test_pooled_patch_stays_poolable_when_rewrapped_onto_a_quantized_module(self):
		layer = _build_int8_patcher().model.diffusion_model.layer
		pooled_patch = lora_patching._create_pooled_stochastic_patch([(_lora_adapter(), 1.0)], 0.04, seed=318008)

		rewrapped, was_wrapped = model_adapter._wrap_existing_int8_patch(layer, pooled_patch, seed=318008)

		self.assertTrue(was_wrapped)
		self.assertIsNot(rewrapped, pooled_patch)
		self.assertTrue(lora_patching._is_pooled_stochastic_patch(rewrapped))


if __name__ == "__main__":
	unittest.main()
