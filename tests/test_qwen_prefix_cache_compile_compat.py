import importlib
import logging
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

import comfy.ldm.qwen_image21.model as qwen_model
import comfy.ldm.wan.model_animate2 as model_animate2
import comfy.memory_management
import comfy.model_management
import comfy.ops


compile_compat = importlib.import_module(f"{PACKAGE_NAME}.qwen_prefix_cache_compile_compat")
lazy_compile = importlib.import_module(f"{PACKAGE_NAME}.int8_lazy_compile")


class _DynamoRecorder(logging.Handler):
	def __init__(self, records):
		super().__init__()
		self.records = records

	def emit(self, record):
		self.records.append(record.getMessage())


class _StubPatcher:
	def get_free_memory(self, device):
		return 8 * 1024 ** 3


def _build_tiny_qwen_model():
	model = qwen_model.QwenImage21Transformer2DModel(
		in_channels=8,
		out_channels=8,
		num_layers=2,
		attention_head_dim=16,
		num_attention_heads=2,
		context_in_dim=32,
		mlp_ratio=3,
		axes_dims_rope=(4, 4, 8),
		dtype=torch.float32,
		device=torch.device("cpu"),
		operations=comfy.ops.disable_weight_init,
	)
	model.requires_grad_(False)
	model.current_patcher = _StubPatcher()
	model.prefix_cache_enabled = True
	return model.eval()


def _install_compile_proxies(model):
	dispatch = torch.compile(lazy_compile._dispatch_compiled_module, backend="eager", dynamic=True)
	for index, block in enumerate(model.transformer_blocks):
		model.transformer_blocks[index] = lazy_compile._CompiledModuleProxy(
			f"diffusion_model.transformer_blocks.{index}",
			block,
			dispatch,
		)


def _pristine_put():
	put = model_animate2.PoseBranchCache.__dict__["put"]
	return getattr(put, "_torchdynamo_orig_callable", put)


class QwenPrefixCacheCompileCompatTests(unittest.TestCase):
	def setUp(self):
		# The shim is process-global; start every test from the unshimmed ComfyUI function.
		model_animate2.PoseBranchCache.put = _pristine_put()
		if hasattr(torch, compile_compat._TORCH_STATE_ATTRIBUTE):
			delattr(torch, compile_compat._TORCH_STATE_ATTRIBUTE)
		torch._dynamo.reset()

	def _run_host_cached_forward(self):
		"""Run two steps of the edit path with the prefix K/V cached in host memory."""
		model = _build_tiny_qwen_model()
		_install_compile_proxies(model)
		x = torch.randn(1, 8, 4, 4)
		context = torch.randn(1, 6, 32)
		ref_latents = [torch.randn(1, 8, 4, 4)]
		transformer_options = {
			"patches": {},
			"patches_replace": {},
			"qwen_image21_cache": {"device": "cpu", "dtype": "default"},
		}

		records = []
		handler = _DynamoRecorder(records)
		logger = logging.getLogger("torch._dynamo")
		logger.addHandler(handler)
		failure = ""
		try:
			with torch.no_grad():
				for _ in range(2):
					model(x, torch.tensor([0.5]), context, ref_latents, [2], transformer_options)
		except Exception as exception:  # Dynamo raises or logs the fake-tensor failure by version
			failure = f"{type(exception).__name__}: {exception}"
		finally:
			logger.removeHandler(handler)
		return failure + "\n".join(records)

	def _pinning_context(self):
		# Real host pinning needs an NVIDIA/AMD build and a pinned-memory budget, so pin
		# the two inputs the upstream failure path depends on and stub the CUDA call.
		cudart = mock.Mock(return_value=mock.Mock(cudaHostRegister=mock.Mock(return_value=0)))
		return [
			mock.patch.object(comfy.model_management, "MAX_PINNED_MEMORY", 1 << 30, create=True),
			mock.patch.object(comfy.model_management, "ensure_pin_registerable", mock.Mock(return_value=True)),
			mock.patch.object(comfy.memory_management, "extra_ram_release", mock.Mock(return_value=0)),
			mock.patch.object(torch.cuda, "cudart", cudart),
		]

	def test_host_cache_write_breaks_compilation_until_the_shim_is_installed(self):
		with mock.patch.object(comfy.model_prefetch, "malloc_graph_begin", mock.Mock()):
			with mock.patch.object(comfy.model_prefetch, "malloc_graph_end", mock.Mock()):
				with mock.patch.object(comfy.model_prefetch, "make_prefetch_queue", mock.Mock(return_value=[])):
					with mock.patch.object(comfy.model_prefetch, "prefetch_queue_pop", mock.Mock()):
						for patch in self._pinning_context():
							patch.start()
							self.addCleanup(patch.stop)

						unshimmed = self._run_host_cached_forward()
						self.assertIn("sizes/strides", unshimmed)

						self.assertTrue(compile_compat.install())
						self.assertTrue(compile_compat.is_installed())
						shimmed = self._run_host_cached_forward()

		self.assertNotIn("sizes/strides", shimmed)
		self.assertTrue(getattr(model_animate2.PoseBranchCache.put, "_torchdynamo_disable", False))
		self.assertGreater(
			len(torch._dynamo.eval_frame._debug_get_cache_entry_list(lazy_compile._dispatch_compiled_module)),
			0,
		)

	def test_install_is_idempotent_within_one_process(self):
		self.assertTrue(compile_compat.install())
		installed_put = model_animate2.PoseBranchCache.put
		self.assertTrue(compile_compat.install())
		self.assertIs(model_animate2.PoseBranchCache.put, installed_put)


if __name__ == "__main__":
	unittest.main()
