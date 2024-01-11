import os
from typing import List
from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler
import torch
from cog import BasePredictor, Input, Path
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
import subprocess

# MODEL_ID refers to a diffusers-compatible model on HuggingFace
# e.g. prompthero/openjourney-v2, wavymulder/Analog-Diffusion, etc
MODEL_ID = "Lykon/dreamshaper-xl-1-0"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=MODEL_CACHE,
            # local_files_only=True,
        )
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            safety_checker=safety_checker,
            cache_dir=MODEL_CACHE,
            # local_files_only=True
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image.",
            choices=[384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1536],
            default=1024,
        ),
        height: int = Input(
            description="Height of output image.",
            choices=[384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1536],
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # check if we have the lora weights downloaded
        # folder_path = "niji-lora"
        # file_path = os.path.join(folder_path, "pytorch_lora_weights.safetensors")
        # if not os.path.exists(file_path):
        #     print("Downloading LoRA weights...")
        #     # download lora with wget from https://f005.backblazeb2.com/file/sd-loras/1990-2.safetensors
        #     os.makedirs(folder_path, exist_ok=True)
        #     download_url = "https://f005.backblazeb2.com/file/sd-loras/1990-2.safetensors"
        #     subprocess.run(["wget", "-O", file_path, download_url])

            # rename the file to pytorch_lora_weights.safetensors
            # os.system("mv ./niji-lora ./pytorch_lora_weights.safetensors")
            
        # self.pipe.load_lora_weights("./niji-lora", weight_name="pytorch_lora_weights.safetensors")
        self.pipe.load_lora_weights("dinocres/niji-lora")

        self.pipe.scheduler = DEISMultistepScheduler.from_config(self.pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
               continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths