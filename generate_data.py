from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import os
import shutil
from config import STEPS, PROMPT, N_IMAGES

shutil.rmtree("images", ignore_errors=True)
os.mkdir("images")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", safety_checker=None
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
print("Disable xformers memory efficient attention")
pipe.unet.set_default_attn_processor()
pipe.vae.set_default_attn_processor()
pipe.disable_xformers_memory_efficient_attention()
pipe.to("cuda")

for i in range(N_IMAGES):
    generator = torch.Generator().manual_seed(i)
    result = pipe(PROMPT, generator=generator, num_inference_steps=STEPS)
    images = result.images
    images[0].save(f"images/{i}.png")
    print(i)
