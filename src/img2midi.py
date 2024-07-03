import numpy as np
import mido
from mido import Message, MidiFile
from PIL import Image

import uuid


def pil_to_midi(src: Image.Image, output_midi_file):
    img_rgb = src.convert("L")
    img_array = np.asarray(img_rgb.point(lambda x: 255 if x > 128 else 0))

    midi = MidiFile()
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120)))
    midi.tracks.append(track)

    print(len(img_array), len(img_array[0]))
    offset = 60
    for height_offset, i in enumerate(img_array):
        for imdata in i:
            track.append(Message("note_on", note=64+height_offset, velocity=100, time=offset+0))
    print(img_array)
    # WIP
