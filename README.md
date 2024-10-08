# image_gen_stable_diff

This project uses the Stable Diffusion model to generate images from text prompts.

## Description

image_gen_stable_diff is a Python-based project that leverages the power of the Stable Diffusion model to create high-quality images based on textual descriptions. It uses the `diffusers` library from Hugging Face to interact with the Stable Diffusion pipeline.

## Features

- Generate images from text prompts
- Utilizes the Stable Diffusion v1.5 model from Runway ML
- Supports CUDA acceleration for faster image generation
- Saves generated images in PNG format

## Requirements

- Python 3.6+
- PyTorch
- diffusers
- Pillow (PIL)
- CUDA-capable GPU (for optimal performance)

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/image_gen_stable_diff.git cd image_gen_stable_diff


2. Install the required packages:

```pip install torch diffusers pillow```


## Usage

1. Open `main.py` in your preferred text editor.
2. Modify the `prompt` variable to describe the image you want to generate.
3. (Optional) Change the `output_path` to specify a different filename or location for the generated image.
4. Run the script:

```python main.py`````

5. The generated image will be saved to the specified output path.

## Example

```python
prompt = "A beautiful landscape with mountains and a lake at sunset"
output_path = "generated_image.png"
generate_image_from_text(prompt, output_path)

This will generate an image of a landscape with mountains and a lake at sunset, and save it as "generated_image.png" in the current directory.

License
[license here]

Contributing
[guidelines for contributing to project]

Acknowledgements
This project uses the Stable Diffusion model by Runway ML.
Thanks to the Hugging Face team for the diffusers library.
