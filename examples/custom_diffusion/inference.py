import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32
).to("cpu")
pipe.unet.load_attn_procs(
    "./model", weight_name="pytorch_custom_diffusion_weights.safetensors"
)
pipe.load_textual_inversion("./model", weight_name="_token_.safetensors")


prompt="a photo of _token_ cat wearing a sunglass"
image = pipe(
    prompt,
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("./outputs/output_cat.png")