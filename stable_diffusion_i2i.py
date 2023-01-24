import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import requests
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

# load the pipeline
device = "cuda"
model_id = "../anything-v4.0"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

init_image = Image.open("../images/image_2950047911628730.png")
width, height = init_image.size
init_image = init_image.resize((768, 768))


prompt = "masterpiece, best quality, 1girl, white hair, medium hair, cat ears, closed eyes, looking at viewer, :3, cute, scarf, jacket, outdoors, streets"
neg_prompt = "bad fingers"
strength = 0.6
guidance_scale = 7.5
num_images_per_prompt = 1
num_inference_steps = 50
num_times = 1

for i in range(num_times):
    generator = torch.Generator(device=device)
    seed = generator.seed()
    generator = generator.manual_seed(seed)

    PipelineOut = pipe(prompt=prompt, negative_prompt=neg_prompt, image=init_image, num_inference_steps=num_inference_steps,
                       strength=strength, guidance_scale=guidance_scale, generator=generator,
                       num_images_per_prompt=num_images_per_prompt)
    images = PipelineOut.images
    print('Generate {}/{} successfully.'.format(str(i), str(num_times)))
    for j in range(num_images_per_prompt):
        images[j].save("../images/image_{}_{}.png".format(str(seed), str(j)))