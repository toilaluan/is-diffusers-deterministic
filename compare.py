from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import os
import shutil
from PIL import Image
from config import STEPS, PROMPT, N_IMAGES
import imagehash

from PIL import Image


def concat_images_horizontally(image1, image2):
    """
    Concatenates two PIL Image objects horizontally.

    :param image1: PIL Image object.
    :param image2: PIL Image object.
    :return: Concatenated PIL Image object.
    """
    # Getting dimensions of the images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Creating a new image with the combined width and the maximum height of the two images
    new_image = Image.new("RGB", (width1 + width2, max(height1, height2)))

    # Pasting the first image on the new image
    new_image.paste(image1, (0, 0))

    # Pasting the second image on the new image
    new_image.paste(image2, (width1, 0))

    return new_image


shutil.rmtree("diffs", ignore_errors=True)
os.mkdir("diffs")
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
    diff = ref_image - image
    if diff != 0:
        concate_image = concat_images_horizontally(ref_image, image)
        concate_image.save(f"diffs/{i}.png")
    print(f"{i}: {ref_hash - hash}")
