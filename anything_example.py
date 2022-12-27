import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_id = "./models/anything-v3.0"
branch_name= "diffusers"
device = torch.device("cuda")

pipe = StableDiffusionPipeline.from_pretrained(model_id, revision=branch_name, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

prompt = "pikachu"
num_inference_steps = 100
height = 768
width = 768
num_images_per_prompt = 1
guidance_scale = 7.5

PipelineOut = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width, num_images_per_prompt=num_images_per_prompt)
images = PipelineOut.images 

print('Generate successfully.')

for i in range(num_images_per_prompt):
    images[i].save("./images/image_{}.png".format(str(i)))
