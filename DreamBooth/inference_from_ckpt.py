import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from accelerate import Accelerator
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import argparse

parser = argparse.ArgumentParser(description="DreamBooth utils")
parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        required=True,
        help="Path to checkpoint of DreamBooth",
    )
parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        required=True,
        help="Path to unwrapped diffusers model",
    )
args = parser.parse_args()

# Load the pipeline with the same arguments (model, revision) that were used for training
# model_id = "/home/models/anything-v4.0"
model_id = args.pretrained_model_name_or_path
pipeline = DiffusionPipeline.from_pretrained(model_id)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
accelerator = Accelerator()

# Use text_encoder if `--train_text_encoder` was used for the initial training
unet, text_encoder = accelerator.prepare(pipeline.unet, pipeline.text_encoder)

# Restore state from a checkpoint path. You have to use the absolute path here.
# accelerator.load_state("/home/output/checkpoint-1000")
accelerator.load_state(args.checkpoint_path)

# Rebuild the pipeline with the unwrapped models (assignment to .unet and .text_encoder should work too)
pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    unet=accelerator.unwrap_model(unet),
    text_encoder=accelerator.unwrap_model(text_encoder),
)

# Perform inference, or save, or push to the hub
# pipeline.save_pretrained("dreambooth_pipeline_1000")
pipeline.save_pretrained(args.save_path)
