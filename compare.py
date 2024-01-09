from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import os
import shutil
from PIL import Image
from config import STEPS, PROMPT, N_IMAGES
import imagehash

# shutil.rmtree("images", ignore_errors=True)
# os.mkdir("images")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")


for i in range(N_IMAGES):
    ref_image = Image.open(f"images/{i}.png")
    generator = torch.manual_seed(i)
    result = pipe(PROMPT, generator=generator, num_inference_steps=STEPS)
    image = result.images[0]
    ref_hash = imagehash.average_hash(ref_image)
    hash = imagehash.average_hash(image)
    print(f"{i}: {ref_hash - hash}")
