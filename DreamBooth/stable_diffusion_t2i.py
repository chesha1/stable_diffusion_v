# pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate scipy
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import argparse

# make speed lower
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser(description="DreamBooth utils")
parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        required=True,
        help="Path to save images",
    )
args = parser.parse_args()

# model_id = "stabilityai/stable-diffusion-2-1"
# model_id = "../stable-diffusion-2-1"
# model_id = "../stable-diffusion-v1-5"
# model_id = "../anything-v4.0"
model_id = args.pretrained_model_name_or_path
device = torch.device("cuda")

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

prompt = "masterpiece, best quality, 1girl, lun, pink and blue hair, purple eyes, looking at viewer, :3, cute, scarf, jacket, outdoors, streets"
neg_prompt = "bad fingers"
num_inference_steps = 50
height = 512
width = 512
num_images_per_prompt = 20
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

    for j in range(num_images_per_prompt):
        # images[j].save("../images/image_{}_{}.png".format(str(seed), str(j)))
        images[j].save(args.save_path.format(str(seed), str(j)))
