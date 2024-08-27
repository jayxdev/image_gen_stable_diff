import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from PIL import Image
import streamlit as st
from io import BytesIO

@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    # Load the model with default settings, which will run on the CPU
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    return pipe

def generate_image_from_text(pipe, prompt, progress_bar):
    def callback(step: int, timestep: int, latents: torch.FloatTensor):
        total_steps = pipe.scheduler.config.num_train_timesteps
        progress = step / total_steps
        progress_bar.progress(progress)

    # Generate the image with progress callback and set callback_steps
    image = pipe(prompt, callback=callback, callback_steps=1).images[0]
    progress_bar.progress(100)  # Ensure progress bar hits 100% at the end
    return image
st.title("Image Generation with Stable Diffusion")

prompt = st.text_input("Enter your prompt:", "A beautiful landscape with mountains and a lake at sunset")
  
if st.button("Generate Image"):
    pipe = load_model()
    progress_bar = st.progress(0)
    
    with st.spinner("Generating image..."):
        image = generate_image_from_text(pipe, prompt, progress_bar)
    
    st.image(image, caption="Generated Image", use_column_width=True)
    
    # Option to download the generated image
    buf = BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="Download Image",
        data=byte_im,
        file_name="generated_image.png",
        mime="image/png"
    )
