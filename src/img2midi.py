import numpy as np
import mido
from mido import Message, MidiFile
from PIL import Image


def count_same(array):
    results = []
    for row in array.copy():
        boundaries = np.flatnonzero(row[:-1] != row[1:]) + 1

        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(row)]])
        lengths = ends - starts
        values = row[starts]

        result = [(values[i], lengths[i]) for i in range(len(lengths))]
        results.append(result)
    return results


def pil_to_midi(src: Image.Image, output_midi_file):
    img_rgb = src.convert("L")
    img_array = np.asarray(img_rgb.point(lambda x: 255 if x > 128 else 0))

    midi = MidiFile()
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120)))
    midi.tracks.append(track)

    print(len(img_array), len(img_array[0]))
    offset = 60
    for height_offset, i in enumerate(count_same(img_array)):
        print(i)
        for imdata in i:
            pass
            # track.append(
            #     Message(
            #         "note_on", note=64 + height_offset, velocity=100, time=offset + 0
            #     )
            # )
    # WIP
