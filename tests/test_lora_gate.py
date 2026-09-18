import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_NAME = Path(__file__).resolve().parents[1].name
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(COMFY_ROOT / "custom_nodes"))

import comfy.model_patcher
import comfy.ops
import comfy.patcher_extension

lora_nodes = importlib.import_module(f"{PACKAGE_NAME}.int8_lora")
lora_dynamic = importlib.import_module(f"{PACKAGE_NAME}.int8_dynamic_lora")
quant = importlib.import_module(f"{PACKAGE_NAME}.int8_quant")
lazy_compile = importlib.import_module(f"{PACKAGE_NAME}.int8_lazy_compile")


class GateDiffusionModel(torch.nn.Module):
	def __init__(self, layer):
		super().__init__()
		self.layer = layer

	def forward(self, x, transformer_options):
		return self.layer(x)


class GateExecutor:
	def __init__(self, diffusion_model):
		self.class_obj = SimpleNamespace(diffusion_model=diffusion_model)

	def __call__(self, x, t, c_concat, c_crossattn, control, transformer_options):
		return self.class_obj.diffusion_model(x, transformer_options=transformer_options)


class QuantizedLoraGateTests(unittest.TestCase):
	def test_gate_preserves_entry_and_can_be_replaced(self):
		entry = lora_nodes.QuantizedLoraSpec("sda.safetensors", -0.75)
		gated = lora_nodes.QuantizedLoraGate.execute(entry, 2)[0]
		self.assertEqual((gated.path, gated.strength, gated.active_steps), (entry.path, -0.75, 2))
		self.assertIsNone(entry.active_steps)
		self.assertEqual(lora_nodes.QuantizedLoraGate.execute(gated, 4)[0].active_steps, 4)
		self.assertEqual(gated.active_steps, 2)
		with self.assertRaises(ValueError):
			lora_nodes.QuantizedLoraGate.execute(entry, -1)

	def test_zero_gate_never_loads_or_patches(self):
		model = object()
		entry = lora_nodes.QuantizedLoraGate.execute(lora_nodes.QuantizedLoraSpec("missing", 1), 0)[0]
		self.assertEqual(lora_nodes.QuantizedLoraPatcher.execute(model, "Stochastic", {"lora_1": entry}), (model,))

	def test_cached_entry_without_schedule_remains_ungated(self):
		entry = lora_nodes.QuantizedLoraSpec("cached.safetensors", 0.75)
		del entry.active_steps
		with mock.patch.object(lora_nodes.INT8LoraLoaderStack, "apply_loras", return_value=("patched",)) as apply:
			self.assertEqual(lora_nodes.QuantizedLoraPatcher.execute("model", "Stochastic", {"lora_1": entry}), ("patched",))
		apply.assert_called_once_with("Stochastic", "model", [("cached.safetensors", 0.75)])

	def test_stack_routes_only_gated_entries_to_runtime(self):
		entries = {
			"lora_1": lora_nodes.QuantizedLoraSpec("style", 0.5),
			"lora_2": lora_nodes.QuantizedLoraSpec("sda", 1, 2),
		}
		with mock.patch.object(lora_nodes.INT8LoraLoaderStack, "apply_loras", return_value=("style_model",)) as ordinary:
			with mock.patch.object(lora_dynamic.INT8DynamicLoraStack, "apply_loras", return_value=("gated_model",)) as runtime:
				result = lora_nodes.QuantizedLoraPatcher.execute("base", "Stochastic", entries)
		ordinary.assert_called_once_with("Stochastic", "base", [("style", 0.5)])
		runtime.assert_called_once_with("style_model", [("sda", 1)], active_steps=[2])
		self.assertEqual(result, ("gated_model",))

	def test_loader_keeps_mixed_precision_patches_runtime_and_parent_unchanged(self):
		model = torch.nn.Module()
		layer = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32).Linear(4, 3, bias=False)
		layer.load_state_dict({"weight": torch.ones(3, 4)})
		self.assertNotIsInstance(layer, torch.nn.Linear)
		model.diffusion_model = GateDiffusionModel(layer)
		patcher = comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
		adapter = quant.LoRAAdapter([], (torch.ones(3, 2), torch.ones(2, 4), None, None, None, None))
		with mock.patch.object(lora_dynamic, "_get_key_map", return_value={}):
			with mock.patch.object(lora_dynamic.folder_paths, "get_full_path", return_value="sda"):
				with mock.patch.object(lora_dynamic.comfy.utils, "load_torch_file", return_value={}):
					with mock.patch.object(lora_dynamic.comfy.lora, "load_lora", return_value={"diffusion_model.layer.weight": adapter}):
						gated = lora_dynamic.INT8DynamicLoraStack().apply_loras(patcher, [("sda", 1)], active_steps=[2])[0]
		self.assertEqual(gated.patches, {})
		self.assertNotIn("dynamic_loras", patcher.model_options["transformer_options"])
		self.assertEqual(gated.model_options["transformer_options"]["dynamic_loras"][0]["active_steps"], 2)
		self.assertEqual(gated.get_attachment("int8_lora_signature")[0], ("Dynamic", "sda", 1.0, ("active_steps", 2)))

	def test_unsupported_adapter_fails_instead_of_becoming_always_on(self):
		model = SimpleNamespace(model=SimpleNamespace(diffusion_model=GateDiffusionModel(torch.nn.Linear(4, 3))))
		for adapter in (object(), quant.LoRAAdapter([], (torch.ones(3, 2), torch.ones(2, 4), None, None, torch.ones(3), None))):
			with self.assertRaisesRegex(ValueError, "ordinary linear LoRA"):
				lora_dynamic._partition_dynamic_patches(model, {"diffusion_model.layer.weight": adapter}, {}, require_runtime=True)

	def test_sigma_gate_cutoff_repeated_evaluations_and_new_runs(self):
		devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
		for device in devices:
			for layer_kind in ("float", "comfy_mixed_float", "toolkit_float", "int8", "w4a4", "w4a8"):
				with self.subTest(device=device, layer=layer_kind):
					if layer_kind == "float":
						layer = torch.nn.Linear(256, 16, bias=False, device=device)
					elif layer_kind == "comfy_mixed_float":
						layer = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32).Linear(256, 16, bias=False, device=device)
						layer.load_state_dict({"weight": torch.randn(16, 256, device=device) * 0.01})
					else:
						layer = quant.Int8TensorwiseOps.Linear(256, 16, bias=False, device=device, dtype=torch.float32)
						weight = torch.randn(16, 256, device=device) * 0.01
						if layer_kind == "w4a4":
							weight = quant.quantize_native_int4(weight)
							layer._quant_format = "convrot_w4a4"
						elif layer_kind == "w4a8":
							weight = quant.quantize_native_w4a8(weight)
							layer._quant_format = quant.W4A8_FORMAT
						elif layer_kind == "int8":
							weight = torch.randint(-8, 9, (16, 256), dtype=torch.int8, device=device)
							layer.weight_scale = torch.tensor(0.01, device=device)
						layer.weight = torch.nn.Parameter(weight, requires_grad=False)
						layer._is_quantized = layer_kind != "toolkit_float"
					model = GateDiffusionModel(layer)
					quant.DynamicLoRAHook.register(model)
					executor = GateExecutor(model)
					x = torch.randn(3, 256, device=device)
					base = model(x, transformer_options={}).detach().clone()
					up, down = torch.randn(16, 2, device=device) * 0.01, torch.randn(2, 256, device=device) * 0.01
					adapter = quant.LoRAAdapter([], (up, down, None, None, None, None))
					style = {"name": "style", "strength": 0.5, "patches": {"diffusion_model.layer.weight": adapter}, "patch_uuid": "style", "active_steps": 8}
					sda = {"name": "sda", "strength": 1.0, "patches": {"diffusion_model.layer.weight": adapter}, "patch_uuid": "sda", "active_steps": 2}
					sigmas = torch.tensor([1.0, 0.9567, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0], device=device)
					opts = {"dynamic_loras": [style, sda], "sample_sigmas": sigmas}
					delta = torch.nn.functional.linear(torch.nn.functional.linear(x, down), up)
					for sigma, strength in ((1.0, 1.5), (1.0, 1.5), (0.9567, 1.5), (0.93, 1.5), (0.9, 0.5), (0.1, 0.5), (1.0, 1.5)):
						actual = lora_dynamic._dynamic_lora_sync_wrapper(executor, x, torch.tensor([sigma], device=device), transformer_options=opts)
						torch.testing.assert_close(actual, base + delta * strength, atol=2e-5, rtol=2e-4)
					self.assertEqual(opts["dynamic_loras"], [style, sda])
					# A different clone without LoRAs must clear every runtime target.
					torch.testing.assert_close(model(x, transformer_options={}), base)

	def test_missing_schedule_fails_explicitly(self):
		with self.assertRaisesRegex(ValueError, "sample_sigmas"):
			lora_dynamic._dynamic_lora_sync_wrapper(None, torch.zeros(1), torch.ones(1), transformer_options={"dynamic_loras": [{"active_steps": 2}]})

	def test_gate_added_after_compilation_applies_runtime_delta(self):
		for layer_kind in ("float", "comfy_mixed_float"):
			with self.subTest(layer=layer_kind):
				torch._dynamo.reset()
				if layer_kind == "float":
					layer = torch.nn.Linear(4, 3, bias=False)
				else:
					layer = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32).Linear(4, 3, bias=False)
				layer.load_state_dict({"weight": torch.ones(3, 4)})
				model = GateDiffusionModel(layer)
				executor = GateExecutor(model)
				graph_count = 0

				def counting_backend(graph, _inputs):
					nonlocal graph_count
					graph_count += 1
					return graph.forward

				compile_wrapper = lazy_compile._make_lazy_compile_wrapper(
					["diffusion_model.layer"],
					{"backend": counting_backend, "fullgraph": True},
					False,
				)
				wrapped = comfy.patcher_extension.WrapperExecutor.new_class_executor(
					executor.__call__, executor.class_obj,
					[lora_dynamic._dynamic_lora_sync_wrapper, compile_wrapper],
				)
				x = torch.ones(1, 4)
				sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0])
				opts = {"sample_sigmas": sigmas}
				base = wrapped.execute(x, torch.ones(1), None, None, None, opts).detach().clone()
				warm_graph_count = graph_count
				adapter = quant.LoRAAdapter([], (torch.ones(3, 1), torch.ones(1, 4), None, None, None, None))
				entry = {"patches": {"diffusion_model.layer.weight": adapter}, "strength": 1.0, "patch_uuid": "late_gate", "active_steps": 5}
				opts["dynamic_loras"] = [entry]
				for sigma in sigmas[:-1]:
					actual = wrapped.execute(x, sigma[None], None, None, None, opts)
					torch.testing.assert_close(actual, base + 4)
				self.assertGreater(graph_count, warm_graph_count)
				active_graph_count = graph_count
				wrapped.execute(x, torch.ones(1), None, None, None, opts)
				self.assertEqual(graph_count, active_graph_count)
				entry["active_steps"] = 2
				actual = wrapped.execute(x, sigmas[2:3], None, None, None, opts)
				torch.testing.assert_close(actual, base)
				actual = wrapped.execute(x, torch.ones(1), None, None, None, opts)
				torch.testing.assert_close(actual, base + 4)

	def test_fused_slices_gate_without_changing_other_outputs(self):
		for offset in ((0, 1, 2), (1, 1, 2)):
			with self.subTest(offset=offset):
				model = GateDiffusionModel(torch.nn.Linear(4, 3, bias=False))
				quant.DynamicLoRAHook.register(model)
				x = torch.randn(2, 4)
				base = model(x, transformer_options={}).detach().clone()
				up = torch.ones(2 if offset[0] == 0 else 3, 1)
				down = torch.ones(1, 2 if offset[0] == 1 else 4)
				adapter = quant.LoRAAdapter([], (up, down, None, None, None, None))
				patches = {("diffusion_model.layer.weight", offset): adapter}
				patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=model))
				dynamic, static = lora_dynamic._partition_dynamic_patches(patcher, patches, {}, require_runtime=True)
				self.assertEqual(static, {})
				opts = {"sample_sigmas": torch.tensor([1.0, 0.8, 0.5, 0]), "dynamic_loras": [{"patches": dynamic, "strength": 1, "patch_uuid": "slice", "active_steps": 2}]}
				expected = base.clone()
				if offset[0] == 0:
					expected[:, 1:3] += x.sum(dim=-1, keepdim=True)
				else:
					expected += x[:, 1:3].sum(dim=-1, keepdim=True)
				executor = GateExecutor(model)
				torch.testing.assert_close(lora_dynamic._dynamic_lora_sync_wrapper(executor, x, torch.ones(1), transformer_options=opts), expected)
				torch.testing.assert_close(lora_dynamic._dynamic_lora_sync_wrapper(executor, x, torch.tensor([0.5]), transformer_options=opts), base)
				with self.assertRaisesRegex(ValueError, "offset"):
					lora_dynamic._partition_dynamic_patches(patcher, {("diffusion_model.layer.weight", (0, 2, 8)): adapter}, {}, require_runtime=True)


if __name__ == "__main__":
	unittest.main()
