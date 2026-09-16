import folder_paths
import comfy.utils
import comfy.lora
import comfy.patcher_extension
import logging
import uuid
import torch

from .int8_quant import _is_linear_like

from .int8_lora_patching import (
    LoRAAdapter,
    _LORA_ADAPTER_AVAILABLE,
    _append_lora_signature,
    _get_key_map,
    _get_supported_quantization_format,
    _model_has_w4a8_modules,
    _resolve_target_module_cached,
    _wrap_static_int8_patches,
    W4A8_QUANTIZATION_FORMAT,
)

_DYNAMIC_LORA_WRAPPER_KEY = "int8_dynamic_lora_sync"


def _is_dynamic_compatible_adapter(adapter):
    if not _LORA_ADAPTER_AVAILABLE or not isinstance(adapter, LoRAAdapter):
        return False

    weights = getattr(adapter, "weights", None)
    if not isinstance(weights, (list, tuple)) or len(weights) < 2:
        return False

    dora_scale = weights[4] if len(weights) > 4 else None
    reshape = weights[5] if len(weights) > 5 else None
    return dora_scale is None and reshape is None


def _partition_dynamic_patches(model_patcher, patch_dict, module_cache, require_runtime=False):
    dynamic_patch_dict = {}
    static_patch_dict = {}

    for key, adapter in patch_dict.items():
        if require_runtime:
            target_module = _resolve_target_module_cached(model_patcher, key, module_cache)
            weight_key = key[0] if isinstance(key, tuple) else key
            if (
                not _is_dynamic_compatible_adapter(adapter)
                or not _is_linear_like(target_module)
                or not weight_key.endswith(".weight")
                or adapter.weights[0].ndim != 2
                or adapter.weights[1].ndim != 2
                or adapter.weights[3] is not None
                or (isinstance(key, tuple) and len(key) > 2)
            ):
                raise ValueError(f"LoRA Gate requires ordinary linear LoRA patches; unsupported target: {weight_key}")
            input_features = target_module.in_features
            output_features = target_module.out_features
            offset = key[1] if isinstance(key, tuple) else None
            if offset is not None:
                if (
                    not isinstance(offset, (tuple, list)) or len(offset) != 3
                    or any(not isinstance(value, int) for value in offset)
                    or offset[0] not in (0, 1) or offset[1] < 0 or offset[2] <= 0
                    or offset[1] + offset[2] > (output_features if offset[0] == 0 else input_features)
                ):
                    raise ValueError(f"LoRA Gate: unsupported linear offset for {weight_key}")
                if offset[0] == 0:
                    output_features = offset[2]
                else:
                    input_features = offset[2]
            up, down = adapter.weights[:2]
            if up.shape[0] != output_features or down.shape[1] != input_features or up.shape[1] != down.shape[0]:
                raise ValueError(f"LoRA Gate: incompatible linear LoRA dimensions for {weight_key}")
            dynamic_patch_dict[key] = adapter
            continue

        if not _is_dynamic_compatible_adapter(adapter):
            static_patch_dict[key] = adapter
            continue

        try:
            target_module = _resolve_target_module_cached(model_patcher, key, module_cache)
            quantization_format = _get_supported_quantization_format(target_module)
        except Exception:
            quantization_format = None

        if quantization_format == W4A8_QUANTIZATION_FORMAT:
            static_patch_dict[key] = adapter
        elif quantization_format is not None:
            dynamic_patch_dict[key] = adapter
        else:
            # Dynamic runtime hooks intentionally cover Toolkit-supported INT8
            # and W4A4 modules only. Preserve LoRA coverage on floating-point and
            # other mixed-precision layers through ComfyUI's ordinary patch path.
            static_patch_dict[key] = adapter

    return dynamic_patch_dict, static_patch_dict


def _warn_if_w4a8_dynamic_fallback(model_patcher):
    if _model_has_w4a8_modules(model_patcher):
        logging.warning(
            "Quantization Toolkit Dynamic LoRA: W4A8 runtime patching is not supported; "
            "affected W4A8 layers will use ComfyUI's Standard LoRA patch path."
        )


def _dynamic_lora_sync_wrapper(executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
    if transformer_options is None:
        transformer_options = {}

    dynamic_loras = transformer_options.get("dynamic_loras", [])
    if any("active_steps" in entry for entry in dynamic_loras):
        sigmas = transformer_options.get("sample_sigmas")
        if sigmas is None or len(sigmas) < 2:
            raise ValueError("LoRA Gate requires a sampler that supplies sample_sigmas in transformer_options.")
        sigma = float(t.flatten()[0])
        if not bool(torch.all(t == t.flatten()[0])):
            raise ValueError("LoRA Gate does not support mixed timesteps in one model batch.")
        transformer_options = transformer_options.copy()
        transformer_options["dynamic_loras"] = [
            entry for entry in dynamic_loras
            if "active_steps" not in entry
            or (entry["active_steps"] > 0 and (
                entry["active_steps"] >= len(sigmas) - 1
                or sigma > float(sigmas[entry["active_steps"]])
            ))
        ]

    base_model = executor.class_obj
    diffusion_model = getattr(base_model, "diffusion_model", None)
    if diffusion_model is not None:
        from .int8_quant import DynamicLoRAHook
        DynamicLoRAHook.sync_from_transformer_options(diffusion_model, transformer_options)

    return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

def _ensure_dynamic_sync_wrapper(model_patcher):
    model_patcher.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, _DYNAMIC_LORA_WRAPPER_KEY)
    model_patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.APPLY_MODEL,
        _DYNAMIC_LORA_WRAPPER_KEY,
        _dynamic_lora_sync_wrapper
    )

class INT8DynamicLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "INT8 diffusion model to receive a runtime dynamic LoRA."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA file from ComfyUI's loras folder."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": "LoRA strength for the diffusion model. Negative values invert the LoRA effect."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    def load_lora(self, model, lora_name, strength):
        if strength == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        model_patcher = model.clone()
        _warn_if_w4a8_dynamic_fallback(model_patcher)
        
        # 1. Get Patch Map
        key_map = _get_key_map(model_patcher)

        patch_dict = comfy.lora.load_lora(lora, key_map, log_missing=True)
        del lora

        dynamic_patch_dict, static_patch_dict = _partition_dynamic_patches(
            model_patcher,
            patch_dict,
            module_cache={},
        )
        del patch_dict

        # 2. Register Global Hook (if not exists)
        from .int8_quant import DynamicLoRAHook
        DynamicLoRAHook.register(model_patcher.model.diffusion_model)
        _ensure_dynamic_sync_wrapper(model_patcher)

        # 3. Add to Dynamic LoRA list in transformer_options
        # This ensures ComfyUI's cloning handles everything and it's non-sticky
        if "transformer_options" not in model_patcher.model_options:
            model_patcher.model_options["transformer_options"] = {}
        else:
            model_patcher.model_options["transformer_options"] = model_patcher.model_options["transformer_options"].copy()
        
        opts = model_patcher.model_options["transformer_options"]
        if "dynamic_loras" not in opts:
            opts["dynamic_loras"] = []
        else:
            # Shallow copy the list to avoid modifying the parent patcher's list
            opts["dynamic_loras"] = opts["dynamic_loras"].copy()
            
        if dynamic_patch_dict:
            opts["dynamic_loras"].append({
                "name": lora_name,
                "strength": strength,
                "patches": dynamic_patch_dict,
                "patch_uuid": uuid.uuid4().hex,
            })

        if static_patch_dict:
            wrapped_static = _wrap_static_int8_patches(model_patcher, static_patch_dict, module_cache={})
            model_patcher.add_patches(wrapped_static, strength)

        _append_lora_signature(model_patcher, "Dynamic", lora_name, strength)
        return (model_patcher,)

class INT8DynamicLoraStack:
    """
    Apply multiple LoRAs in one node for efficiency.
    """
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {"model": ("MODEL", {"tooltip": "INT8 diffusion model to receive runtime dynamic LoRAs."})},
            "optional": {},
        }
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        for i in range(1, 11):
            inputs["optional"][f"lora_{i}"] = (lora_list, {"tooltip": f"Optional dynamic LoRA slot {i}. Choose None to leave this slot unused."})
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": f"Strength for dynamic LoRA slot {i}. Ignored when the slot is None or strength is 0."})
        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_stack"
    CATEGORY = "loaders"

    def apply_stack(self, model, **kwargs):
        lora_entries = []
        for i in range(1, 11):
            lora_name = kwargs.get(f"lora_{i}")
            strength = kwargs.get(f"strength_{i}", 0)
            if lora_name and lora_name != "None" and strength != 0:
                lora_entries.append((lora_name, strength))

        return self.apply_loras(model, lora_entries)

    def apply_loras(self, model, lora_entries, active_steps=None):

        if not lora_entries:
            return (model,)

        model_patcher = model.clone()
        if active_steps is None:
            _warn_if_w4a8_dynamic_fallback(model_patcher)

        key_map = _get_key_map(model_patcher)

        from .int8_quant import DynamicLoRAHook
        DynamicLoRAHook.register(model_patcher.model.diffusion_model)
        _ensure_dynamic_sync_wrapper(model_patcher)

        if "transformer_options" not in model_patcher.model_options:
            model_patcher.model_options["transformer_options"] = {}
        else:
            model_patcher.model_options["transformer_options"] = model_patcher.model_options["transformer_options"].copy()

        opts = model_patcher.model_options["transformer_options"]
        existing_loras = opts.get("dynamic_loras", [])
        opts["dynamic_loras"] = existing_loras.copy()
        module_cache = {}
        dynamic_patch_count = 0
        static_patch_count = 0

        for index, (lora_name, strength) in enumerate(lora_entries):
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
            patch_dict = comfy.lora.load_lora(lora_data, key_map, log_missing=True)
            del lora_data

            if active_steps is not None and not patch_dict:
                raise ValueError(f"LoRA Gate: no model patches matched {lora_name}.")
            dynamic_patch_dict, static_patch_dict = _partition_dynamic_patches(
                model_patcher,
                patch_dict,
                module_cache,
                require_runtime=active_steps is not None,
            )
            del patch_dict
            dynamic_patch_count += len(dynamic_patch_dict)
            static_patch_count += len(static_patch_dict)

            if dynamic_patch_dict:
                opts["dynamic_loras"].append({
                    "name": lora_name,
                    "strength": strength,
                    "patches": dynamic_patch_dict,
                    "patch_uuid": uuid.uuid4().hex,
                })
                if active_steps is not None:
                    opts["dynamic_loras"][-1]["active_steps"] = active_steps[index]

            if static_patch_dict:
                wrapped_static = _wrap_static_int8_patches(
                    model_patcher,
                    static_patch_dict,
                    module_cache=module_cache,
                )
                model_patcher.add_patches(wrapped_static, strength)

            _append_lora_signature(
                model_patcher, "Dynamic", lora_name, strength,
                active_steps=None if active_steps is None else active_steps[index],
            )

        logging.info(
            f"Quantization Toolkit LoRA stack (Dynamic): loaded {len(lora_entries)} LoRAs in a single pass "
            f"(runtime patch targets={dynamic_patch_count}, standard fallback targets={static_patch_count})."
        )
        return (model_patcher,)

NODE_CLASS_MAPPINGS = {
    "INT8DynamicLoraLoader": INT8DynamicLoraLoader,
    "INT8DynamicLoraStack": INT8DynamicLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INT8DynamicLoraLoader": "Load LoRA (Quantized, Dynamic)",
    "INT8DynamicLoraStack": "Load LoRA Stack (Quantized, Dynamic)",
}
