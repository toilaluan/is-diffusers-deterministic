from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import os
import shutil
from PIL import Image
from config import STEPS, PROMPT, N_IMAGES
import imagehash
from PIL import Image
from typing import List
import timm
import torchvision.transforms as T
import torch.nn.functional as F


def get_transform(model_name):
    data_config = timm.get_pretrained_cfg(model_name).to_dict()
    mean = data_config["mean"]
    std = data_config["std"]
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return transform


MODEL = timm.create_model("resnet50", pretrained=True, num_classes=0)
MODEL.to("cuda")
MODEL.eval()
TRANSFORM = get_transform("resnet50")
THRESHOLD = 0.95


def get_similarity(image_1, image_2):
    image_1 = TRANSFORM(image_1).unsqueeze(0).to("cuda")
    image_2 = TRANSFORM(image_2).unsqueeze(0).to("cuda")
    prob = F.cosine_similarity(MODEL(image_1), MODEL(image_2))
    print("Prob:", prob.item(), flush=True)
    return prob.item()


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
    "runwayml/stable-diffusion-v1-5", safety_checker=None
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.unet.set_default_attn_processor()
pipe.vae.set_default_attn_processor()
pipe.disable_xformers_memory_efficient_attention()
pipe.to("cuda")


for i in range(N_IMAGES):
    ref_image = Image.open(f"images/{i}.png")
    generator = torch.manual_seed(i)
    result = pipe(PROMPT, generator=generator, num_inference_steps=STEPS)
    image = result.images[0]
    sim = get_similarity(ref_image, image)
    ref_hash = imagehash.average_hash(ref_image, hash_size=16)
    hash = imagehash.average_hash(image, hash_size=16)
    diff = ref_hash - hash
    if diff != 0:
        concate_image = concat_images_horizontally(ref_image, image)
        concate_image.save(f"diffs/{i}.png")
    print(f"{i}: {ref_hash - hash}")
