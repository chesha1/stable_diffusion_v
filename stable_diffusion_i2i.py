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
model_id = "/home/models/model4"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

init_image = Image.open("/home/images/test.png")
width, height = init_image.size
# init_image = init_image.resize((768, 768))


prompt = "((masterpiece)), (((best quality))), ((ultra-detailed)), ((illustration)), stage, concert, a lot of waving glow sticks, performance, audience, lights, ((1 girl)), (((((lun))))), pink hair, light green eyes, (hands behind back), (solo), ((singing)), ((leaning_forward)), (arms_behind_back), ((extremely_detailed_eyes_and_face)), colorful, Tokyo Dome, ray tracing"
neg_prompt = "2 girl, an incomplete face, a contorted face, bad anatomy, ((bad hands)), lowres, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, (missing fingers), bad hands, long neck, Humpbacked, extra legs"

strength = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
guidance_scale = 7.5
num_images_per_prompt = 3
num_inference_steps = 50
num_times = 4

for st in strength:
    for i in range(num_times):
        generator = torch.Generator(device=device)
        seed = generator.seed()
        generator = generator.manual_seed(seed)

        PipelineOut = pipe(prompt=prompt, negative_prompt=neg_prompt, image=init_image, num_inference_steps=num_inference_steps,
                           strength=st, guidance_scale=guidance_scale, generator=generator,
                           num_images_per_prompt=num_images_per_prompt)
        images = PipelineOut.images
        print('Generate {}/{} successfully.'.format(str(i+1), str(num_times)))
        for j in range(num_images_per_prompt):
            images[j].save("../images/image_{}_{}_{}.png".format(st, str(seed), str(j)))
    print('Generate strength {} successfully.'.format(st))