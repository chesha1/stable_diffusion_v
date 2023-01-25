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
# model_id = "../stable-diffusion-2-1"
# model_id = "../stable-diffusion-v1-5"
model_id = "/home/models/model4"
device = torch.device("cuda")

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

prompt = "masterpiece, best quality, detailed, ultra-detailed, 1girl, young girl, loli, small breasts, lun, white and pink hair, detailed anime face, beautiful detailed face, green eyes, beautiful detailed eyes, looking at viewer, :3, cute, outdoors, streets, artistic expression, dynamic expression, illustrative expression, detailed expression, ultra-detailed expression, floating expression, painted expression"
neg_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
num_inference_steps = 50
height = 768
width = 512
num_images_per_prompt = 5
guidance_scale = 7.5
num_times = 1
img_index = 0

for i in range(num_times):
    generator = torch.Generator(device=device)
    latents = None
    seeds = []
    for _ in range(num_images_per_prompt):
        seed = generator.seed()
        seeds.append(seed)
        generator = generator.manual_seed(seed)

        image_latents = torch.randn(
            (1, pipe.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=device,
            dtype=torch.float16
        )
        latents = image_latents if latents is None else torch.cat((latents, image_latents))

    PipelineOut = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                       height=height, width=width, num_images_per_prompt=num_images_per_prompt, latents=latents,
                       negative_prompt=neg_prompt)
    images = PipelineOut.images

    print('Generate {}/{} successfully.'.format(i + 1, num_times))

    for j in range(num_images_per_prompt):
        images[j].save("../images/image_{}_{}.png".format(img_index, seeds[j]))
        img_index = img_index + 1
