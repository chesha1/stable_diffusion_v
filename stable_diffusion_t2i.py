# pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate scipy
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# make speed lower
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True

# model_id = "stabilityai/stable-diffusion-2-1"
model_id = "../stable-diffusion-v1-5"
device = torch.device("cuda")

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

prompt = "anime girl, beautiful, big breast, blue eyes, long hair, masterpiece"
neg_prompt = "bad fingers"
num_inference_steps = 50
height = 256
width = 256
num_images_per_prompt = 3
guidance_scale = 7.5
num_times = 1

for i in range(num_times):
    generator = torch.Generator(device=device)
    seed = generator.seed()
    generator = generator.manual_seed(seed)

    PipelineOut = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                       height=height, width=width, num_images_per_prompt=num_images_per_prompt, generator=generator,
                       negative_prompt=neg_prompt)
    images = PipelineOut.images

    print('Generate {} successfully.'.format(str(i)))

    for i in range(num_images_per_prompt):
        images[i].save("../images/image_{}_{}.png".format(str(seed), str(i)))
