from PIL import Image
import numpy as np


def pil_to_midi(src: Image.Image, output_midi_file):
    img_rgb = src.convert("RGB")
    imgarray = np.asarray(img_rgb)
    midi_data = []

    for i in imgarray:
        for j in i:
            pass
    print(imgarray)
    # WIP
