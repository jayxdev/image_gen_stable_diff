
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_image_from_text(prompt, output_path):
    # Load the Stable Diffusion model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Generate the image
    image = pipe(prompt).images[0]

    # Save the image
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    prompt = "A beautiful landscape with mountains and a lake at sunset"
    output_path = "generated_image.png"
    generate_image_from_text(prompt, output_path)
