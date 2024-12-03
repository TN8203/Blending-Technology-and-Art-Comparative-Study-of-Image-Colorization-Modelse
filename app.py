import torch
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import gradio as gr

import transformers
transformers.utils.move_cache()

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from accelerate import Accelerator

import warnings
warnings.filterwarnings("ignore") 


from models import MainModel, UNetAuto, Autoencoder  
from utils import lab_to_rgb, build_res_unet, build_mobilenet_unet  # Utility to convert LAB to RGB
from stable import blip_image_captioning, apply_color

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Stable diffusion

accelerator = Accelerator(
    mixed_precision="fp16"
)

controlnet = ControlNetModel.from_pretrained(
    pretrained_model_name_or_path="nickpai/sdxl_light_caption_output",
    subfolder="checkpoint-30000/controlnet",
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet
)
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large",
)
blip_generator = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large",
)

pipe.to(accelerator.device)
blip_generator.to(accelerator.device)

def colorize_single_image(image, positive_prompt, negative_prompt, caption_generate):
    image = PIL.Image.fromarray(image)

    torch.cuda.empty_cache()
    if caption_generate:
        caption = blip_image_captioning(image=image, device=accelerator.device, processor=blip_processor, generator=blip_generator)
    else:
        caption = ""

    original_size = image.size
    control_image = image.convert("L").convert("RGB").resize((512, 512))
    prompt = [positive_prompt + ", " + caption]
    
    colorized_image = pipe(prompt=prompt,
                           num_inference_steps=5, 
                           generator=torch.manual_seed(0),
                           image=control_image,
                           negative_prompt=negative_prompt).images[0]
    result_image = apply_color(control_image, colorized_image)
    result_image = result_image.resize(original_size)
    return result_image, caption if caption_generate else gr.update(visible=False)

# Hàm load models cho autoencoder và gan
def load_autoencoder_model(auto_model_path):
    unet = UNetAuto(in_channels=1, out_channels=2).to(device)
    model = Autoencoder(unet).to(device)
    model.load_state_dict(torch.load(auto_model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_model(generator_model_path, colorization_model_path, model_type='resnet'):
    if model_type == 'resnet':
        net_G = build_res_unet(n_input=1, n_output=2, size=256)
    elif model_type == 'mobilenet':
        net_G = build_mobilenet_unet(n_input=1, n_output=2, size=256)
    
    net_G.load_state_dict(torch.load(generator_model_path, map_location=device))
    model = MainModel(net_G=net_G)
    model.load_state_dict(torch.load(colorization_model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

resnet_model = load_model(
    "weight/pascal_res18-unet.pt",
    "weight/pascal_final_model_weights.pt",
    model_type='resnet'
)

mobilenet_model = load_model(
    "weight/mobile-unet.pt",
    "weight/mobile_pascal_final_model_weights.pt",
    model_type='mobilenet'
)

autoencoder_model = load_autoencoder_model("weight/autoencoder.pt")

# Transformations
def preprocess_image(image):
    image = image.resize((256, 256))
    image = transforms.ToTensor()(image)[:1] * 2. - 1.
    return image

def postprocess_image(grayscale, prediction, original_size):
    # Convert Lab back to RGB and resize to the original image size
    colorized_image = lab_to_rgb(grayscale.unsqueeze(0), prediction.cpu())[0]
    colorized_image = Image.fromarray((colorized_image * 255).astype("uint8"))
    return colorized_image.resize(original_size)

# Prediction function with output control
def colorize_image(input_image, mode):
    grayscale_image = Image.fromarray(input_image).convert('L')
    original_size = grayscale_image.size  # Store original size
    grayscale = preprocess_image(grayscale_image).to(device)
    
    with torch.no_grad():
        resnet_output = resnet_model.net_G(grayscale.unsqueeze(0))
        mobilenet_output = mobilenet_model.net_G(grayscale.unsqueeze(0))
        autoencoder_output = autoencoder_model(grayscale.unsqueeze(0))
    
    # Resize outputs to match the original size
    resnet_colorized = postprocess_image(grayscale, resnet_output, original_size)
    mobilenet_colorized = postprocess_image(grayscale, mobilenet_output, original_size)
    autoencoder_colorized = postprocess_image(grayscale, autoencoder_output, original_size)
    
    if mode == "ResNet":
        return resnet_colorized, None, None
    elif mode == "MobileNet":
        return None, mobilenet_colorized, None
    elif mode == "Unet":
        return None, None, autoencoder_colorized
    elif mode == "Comparison":
        return resnet_colorized, mobilenet_colorized, autoencoder_colorized


def gradio_interface():
    with gr.Blocks() as app:
        with gr.Tab("Mode Colorization no Prompting"):
            with gr.Blocks():
                input_image = gr.Image(type="numpy", label="Upload an Image")
                output_modes = gr.Radio(
                    choices=["ResNet", "MobileNet", "Unet", "Comparison"],
                    value="ResNet",
                    label="Output Mode"
                )
                
                submit_button = gr.Button("Submit")

                with gr.Row():  # Place output images in a single row
                    resnet_output = gr.Image(label="Colorized Image (ResNet18)", visible=False)
                    mobilenet_output = gr.Image(label="Colorized Image (MobileNet)", visible=False)
                    autoencoder_output = gr.Image(label="Colorized Image (Unet)", visible=False)

                def update_visibility(mode):
                    if mode == "ResNet":
                        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                    elif mode == "MobileNet":
                        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                    elif mode == "Unet":
                        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
                    elif mode == "Comparison":
                        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

                output_modes.change(
                    fn=update_visibility,
                    inputs=[output_modes],
                    outputs=[resnet_output, mobilenet_output, autoencoder_output]
                )

                submit_button.click(
                    fn=colorize_image,
                    inputs=[input_image, output_modes],
                    outputs=[resnet_output, mobilenet_output, autoencoder_output]
                )

        with gr.Tab("Stable Diffusion"):
            with gr.Blocks():
                sd_image = gr.Image(label="Upload a Color Image")
                positive_prompt = gr.Textbox(label="Positive Prompt", placeholder="Text for positive prompt")
                negative_prompt = gr.Textbox(
                    value="low quality, bad quality, low contrast, black and white, bw, monochrome, grainy, blurry, historical, restored, desaturate",
                    label="Negative Prompt", placeholder="Text for negative prompt"
                )
                generate_caption = gr.Checkbox(label="Generate Caption?", value=False)
                submit_sd = gr.Button("Generate")

                sd_output_image = gr.Image(label="Colorized Image")
                sd_caption = gr.Textbox(label="Captioning Result", show_copy_button=True, visible=False)

                submit_sd.click(
                    fn=colorize_single_image,
                    inputs=[sd_image, positive_prompt, negative_prompt, generate_caption],
                    outputs=[sd_output_image, sd_caption]
                )

    return app




# Launch
if __name__ == "__main__":
    gradio_interface().launch()
