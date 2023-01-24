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
model_id = "/home/models/mo-di-diffusion"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

init_image = Image.open("/data/pink_cleaned_processed/1.png")
width, height = init_image.size
# init_image = init_image.resize((512, 512))


prompt = "beautiful anime girl, CG render, 8k, highly detailed, digital painting, masterpiece, best quality, 1girl, pink hair, green eyes, stars around head, looking at viewer, modern disney style"
neg_prompt = "bad fingers"
strength = 0.5
guidance_scale = 7.5
num_images_per_prompt = 1
num_inference_steps = 80
num_times = 15



def generate_img(self, stren, gui):
    guidance_scale = gui
    strength = stren
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
            images[j].save("../images/image_{}_{}_{}.png".format(strength, guidance_scale, str(seed)))





stren_list = [0.5, 0.6, 0.7, 0.8, 0.9]
gui_list = [7.5, 8.0, 8.5, 9.0, 9.5]

for i in stren_list:
    for j in gui_list:
        generate_img(stren=i, gui=j)







