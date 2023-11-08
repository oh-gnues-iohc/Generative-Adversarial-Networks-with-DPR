from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor, ToPILImage
import torch
import os

FONT = ImageFont.truetype("timesbd.ttf", 10)
TO_PIL = ToPILImage()


def draw_examples(input: torch.Tensor, prompt: str, path: str, name: str):
    img = TO_PIL(input)
    canvas = Image.new('RGB', (300, 300), color='white')
    x_offset = (300 - 224) // 2
    y_offset = (300 - 224) // 2
    canvas.paste(img, (x_offset, y_offset))
    draw = ImageDraw.Draw(canvas)
    text_width = draw.textlength(prompt, FONT)
    text_x = (300 - text_width) // 2
    text_y = y_offset + img.height + 10

    draw.text((text_x, text_y), prompt, fill='black', font=FONT)

    canvas.save(os.path.join(path, f"{name}.jpg"))