# Checkpoint Sources

Pre-quantized checkpoints avoid local conversion time. On-the-fly quantization remains more flexible but requires loading the source weights and converting them locally.

## Vistralis

| Model | Link |
| --- | --- |
| FLUX.2-klein-base-9b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-base-9b-INT8-transformer) |
| FLUX.2-klein-base-4b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-base-4b-INT8-transformer) |
| FLUX.2-klein-9b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-9b-INT8-transformer) |
| FLUX.2-klein-4b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-4b-INT8-transformer) |

## Additional Checkpoints

| Model | Link |
| --- | --- |
| Chroma1-HD | [Download](https://huggingface.co/bertbobson/Chroma1-HD-INT8Tensorwise) |
| Z-Image-Turbo | [Download](https://huggingface.co/bertbobson/Z-Image-Turbo-INT8-Tensorwise) |
| Anima | [Download](https://huggingface.co/bertbobson/Anima-INT8-QUIP) |

## Grouped INT8

| Model | Format | Link |
| --- | --- | --- |
| MOSS-Image SFT | Native ConvRot W8A8 | [Download](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_sft_int8_convrot.safetensors) |
| MiniMax H3 FL2VA pruned | Native W6A8 | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_w6a8.safetensors) |
| MiniMax H3 FL2VA pruned | Legacy-metadata mixed W4A8 | [Download](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors) |

The Kijai `w4a8_mixed` file uses legacy `_quantization_metadata` and stores its transformer blocks as Comfy's native `asym_w4a8_int8` format. It is not a separate kernel format.
