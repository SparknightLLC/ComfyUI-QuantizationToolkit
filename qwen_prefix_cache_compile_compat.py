"""Temporary torch.compile compatibility boundary for the Qwen Image 2.1 prefix cache.

Qwen Image 2.1 edit runs keep the text and reference K/V in a cross-step cache.
When that cache lives in host memory, ``PoseBranchCache.put`` pins the stacked K/V
through ``comfy.model_management.pin_memory``, which reads ``tensor.nbytes``. That
read has no symbolic-shape implementation, so TorchDynamo fails to convert the
frame with "Cannot call numel() on tensor with symbolic sizes/strides" and prints a
TorchRuntimeError traceback for every transformer block it tries to compile.

The cache write is a host-memory side effect, not part of the attention math, so the
Toolkit marks ``PoseBranchCache.put`` as a Dynamo graph break: the write runs eagerly
and the surrounding block graph still compiles.

Retirement path:
1. ComfyUI makes the cache write traceable, either by sizing the pin from static
   shapes or by moving the write out of the compiled block call path.
2. The Qwen warning in ``int8_lazy_compile`` and this module can then be removed.

This is a runtime monkeypatch of a ComfyUI internal, which is why it stays small,
separate, and reversible by deleting the module. It is never installed when the
stock ``Qwen Image 2.1 Cache`` node selects ``gpu`` or ``off``, because those
settings cannot reach the host pinning path. ``PoseBranchCache`` is shared with
Wan Animate 2, so this shim also applies to that cache when it is active.

Evidence basis: ComfyUI 0.37.0 with torch 2.11.0+cu130 on Windows, RTX 3090. An
unshimmed run fails at ``model_management.py`` line 1660 with "Cannot call numel()
on tensor with symbolic sizes/strides"; the shimmed run compiles the blocks and
still pins the host cache. Revalidate against the installed ComfyUI before keeping
this shim, and drop it once the cache write is traceable.
"""

import logging

import torch


_SHIM_REVISION = 1
_TORCH_STATE_ATTRIBUTE = "_comfyui_quantization_toolkit_qwen_prefix_cache_compile_compat"

try:
	_compile_disable = torch.compiler.disable
except AttributeError:
	_compile_disable = getattr(torch._dynamo, "disable", None)


def _get_pose_branch_cache_class():
	try:
		from comfy.ldm.wan.model_animate2 import PoseBranchCache
	except Exception:
		return None
	return PoseBranchCache


def _record_state(installed, error):
	setattr(
		torch,
		_TORCH_STATE_ATTRIBUTE,
		{
			"revision": _SHIM_REVISION,
			"installed": installed,
			"error": error,
		},
	)


def is_available():
	return _compile_disable is not None and _get_pose_branch_cache_class() is not None


def get_install_error():
	state = getattr(torch, _TORCH_STATE_ATTRIBUTE, None)
	if isinstance(state, dict) and state.get("installed"):
		return None
	if isinstance(state, dict) and state.get("revision") == _SHIM_REVISION:
		return state.get("error")
	return "The Qwen prefix cache compile shim has not been installed."


def install():
	"""Mark the prefix cache write as a graph break. Idempotent for one process."""
	state = getattr(torch, _TORCH_STATE_ATTRIBUTE, None)
	if isinstance(state, dict) and state.get("revision") == _SHIM_REVISION:
		if state.get("installed"):
			return True
		return False

	cache_class = _get_pose_branch_cache_class()
	if _compile_disable is None or cache_class is None:
		error = (
			"torch.compiler.disable or comfy.ldm.wan.model_animate2.PoseBranchCache "
			"is unavailable in this environment."
		)
		_record_state(False, error)
		return False

	if "put" not in cache_class.__dict__:
		error = "PoseBranchCache.put is inherited; refusing to patch an unknown owner."
		_record_state(False, error)
		return False

	try:
		cache_class.put = _compile_disable(cache_class.__dict__["put"])
	except Exception as exception:
		_record_state(False, f"{type(exception).__name__}: {exception}")
		return False

	_record_state(True, None)
	return True


def is_installed():
	state = getattr(torch, _TORCH_STATE_ATTRIBUTE, None)
	return bool(isinstance(state, dict) and state.get("installed"))
