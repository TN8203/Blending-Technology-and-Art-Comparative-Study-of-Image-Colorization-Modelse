# for image captioning
import PIL
import torch
from torchvision import transforms

import transformers
transformers.utils.move_cache()

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from accelerate import Accelerator

def remove_unlikely_words(prompt: str) -> str:
    """
    Removes unlikely words from a prompt.

    Args:
        prompt: The text prompt to be cleaned.

    Returns:
        The cleaned prompt with unlikely words removed.
    """
    unlikely_words = []

    a1_list = [f'{i}s' for i in range(1900, 2000)]
    a2_list = [f'{i}' for i in range(1900, 2000)]
    a3_list = [f'year {i}' for i in range(1900, 2000)]
    a4_list = [f'circa {i}' for i in range(1900, 2000)]
    b1_list = [f"{year[0]} {year[1]} {year[2]} {year[3]} s" for year in a1_list]
    b2_list = [f"{year[0]} {year[1]} {year[2]} {year[3]}" for year in a1_list]
    b3_list = [f"year {year[0]} {year[1]} {year[2]} {year[3]}" for year in a1_list]
    b4_list = [f"circa {year[0]} {year[1]} {year[2]} {year[3]}" for year in a1_list]

    words_list = [
        "black and white,", "black and white", "black & white,", "black & white", "circa", 
        "balck and white,", "monochrome,", "black-and-white,", "black-and-white photography,", 
        "black - and - white photography,", "monochrome bw,", "black white,", "black an white,",
        "grainy footage,", "grainy footage", "grainy photo,", "grainy photo", "b&w photo",
        "back and white", "back and white,", "monochrome contrast", "monochrome", "grainy",
        "grainy photograph,", "grainy photograph", "low contrast,", "low contrast", "b & w",
        "grainy black-and-white photo,", "bw", "bw,",  "grainy black-and-white photo",
        "b & w,", "b&w,", "b&w!,", "b&w", "black - and - white,", "bw photo,", "grainy  photo,",
        "black-and-white photo,", "black-and-white photo", "black - and - white photography",
        "b&w photo,", "monochromatic photo,", "grainy monochrome photo,", "monochromatic",
        "blurry photo,", "blurry,", "blurry photography,", "monochromatic photo",
        "black - and - white photograph,", "black - and - white photograph", "black on white,",
        "black on white", "black-and-white", "historical image,", "historical picture,", 
        "historical photo,", "historical photograph,", "archival photo,", "taken in the early",
        "taken in the late", "taken in the", "historic photograph,", "restored,", "restored", 
        "historical photo", "historical setting,",
        "historic photo,", "historic", "desaturated!!,", "desaturated!,", "desaturated,", "desaturated", 
        "taken in", "shot on leica", "shot on leica sl2", "sl2",
        "taken with a leica camera", "taken with a leica camera", "leica sl2", "leica", "setting", 
        "overcast day", "overcast weather", "slight overcast", "overcast", 
        "picture taken in", "photo taken in", 
        ", photo", ",  photo", ",   photo", ",    photo", ", photograph",
        ",,", ",,,", ",,,,", " ,", "  ,", "   ,", "    ,", 
    ]

    unlikely_words.extend(a1_list)
    unlikely_words.extend(a2_list)
    unlikely_words.extend(a3_list)
    unlikely_words.extend(a4_list)
    unlikely_words.extend(b1_list)
    unlikely_words.extend(b2_list)
    unlikely_words.extend(b3_list)
    unlikely_words.extend(b4_list)
    unlikely_words.extend(words_list)
    
    for word in unlikely_words:
        prompt = prompt.replace(word, "")
    return prompt

def blip_image_captioning(image, device, processor, generator, conditional="a photography of"):
    # Load the processor and model
    if processor is None:
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
    if generator is None:
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=torch.float16
        ).to(device)

    # Prepare inputs
    inputs = processor(
        image, 
        text=conditional, 
        return_tensors="pt"
    ).to(device)
    
    # Generate the caption
    out = generator.generate(**inputs, max_new_tokens=20)  # Use max_new_tokens for better clarity
    caption = processor.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    caption = remove_unlikely_words(caption)

    return caption

def apply_color(image: PIL.Image.Image, color_map: PIL.Image.Image) -> PIL.Image.Image:
    # Convert input images to LAB color space
    image_lab = image.convert('LAB')
    color_map_lab = color_map.convert('LAB')

    # Split LAB channels
    l, a , b = image_lab.split()
    _, a_map, b_map = color_map_lab.split()

    # Merge LAB channels with color map
    merged_lab = PIL.Image.merge('LAB', (l, a_map, b_map))

    # Convert merged LAB image back to RGB color space
    result_rgb = merged_lab.convert('RGB')
    return result_rgb