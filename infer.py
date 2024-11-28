import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import gradio as gr

from models import MainModel  # Import class for your main model
from utils import lab_to_rgb, build_res_unet#, build_mobile_unet  # Utility to convert LAB to RGB

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(generator_model_path, colorization_model_path): #, model_type='resnet')

    #if model_type == 'resnet':
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    # elif model_type == 'mobilenet':
    #     net_G = build_mobile_unet(n_input=1, n_output=2, size=256)
    
    net_G.load_state_dict(torch.load(generator_model_path, map_location=device))
    
    # Create MainModel and load weights
    model = MainModel(net_G=net_G)
    model.load_state_dict(torch.load(colorization_model_path, map_location=device))
    
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()
    
    return model

# Load pretrained models
resnet_model = load_model(
    "weight/pascal_res18-unet.pt",
    "weight/pascal_final_model_weights.pt"
    # model_type='resnet'
)

# mobilenet_model = load_model(
#     "weight/mobile-unet.pt",
#     "weight/mobile_pascal_final_model_weights.pt",
#     model_type='mobilenet'
# )

# Transformations
def preprocess_image(image):
    image = image.resize((256, 256))
    image = transforms.ToTensor()(image)[:1] * 2. - 1.  # Normalize to [-1, 1]
    return image

def postprocess_image(grayscale, prediction):
    return lab_to_rgb(grayscale.unsqueeze(0), prediction.cpu())[0]

# Prediction function
def colorize_image(input_image):
    # Convert input to grayscale
    input_image = Image.fromarray(input_image).convert('L')
    grayscale = preprocess_image(input_image).to(device)
    
    # Generate predictions
    with torch.no_grad():
        resnet_output = resnet_model.net_G(grayscale.unsqueeze(0))
        # mobilenet_output = mobilenet_model.net_G(grayscale.unsqueeze(0))
    
    # Post-process results
    resnet_colorized = postprocess_image(grayscale, resnet_output)
    # mobilenet_colorized = postprocess_image(grayscale, mobilenet_output)
    
    return (
        input_image,  # Grayscale image
        resnet_colorized  # ResNet18 colorized image
        # mobilenet_colorized  # MobileNet colorized image
    )

# Gradio Interface
interface = gr.Interface(
    fn=colorize_image,
    inputs=gr.Image(type="numpy", label="Upload a Color Image"),
    outputs=[
        gr.Image(label="Grayscale Image"),
        gr.Image(label="Colorized Image (ResNet18)")
        # gr.Image(label="Colorized Image (MobileNet)")
    ],
    title="Image Colorization",
    description="Upload a color image"
)

# Launch Gradio app
if __name__ == '__main__':
    interface.launch()