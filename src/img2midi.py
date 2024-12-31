import numpy as np
import mido
from mido import Message, MidiFile
from PIL import Image
from typing import Literal


def count_same(array):
    results = []
    for row in array:
        boundaries = np.flatnonzero(row[:-1] != row[1:]) + 1

        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(row)]])
        lengths = (ends - starts).astype(np.int64)
        values = row[starts]

        result = [[int(values[i]), int(lengths[i])] for i in range(len(lengths))]
        results.append(result)
    return results


def pil_to_midi(src: Image.Image, output_midi_file):
    img_rgb = src.convert("L")
    img_array = np.asarray(img_rgb.point(lambda x: 255 if x > 128 else 0))

    midi = MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120)))

    print(len(img_array), len(img_array[0]))
    counted = count_same(img_array)
    offset = 120

    info: list[tuple[Literal[0, 1], int, int]] = []
    for height_offset, i in enumerate(counted):
        now = 0
        for im_data in i:
            if im_data[0] == 0:
                o = offset - height_offset
                info.append((0, o, now))
                info.append((1, o, now + im_data[1]))
            now += im_data[1]
    print(info)
    info.sort(key=lambda x: x[2])
    ooooo = 0
    for opcode, note, t in info:
        aa = t * 15 - ooooo
        msg_type = "note_on" if opcode == 0 else "note_off"
        track.append(Message(msg_type, note=note, velocity=100, time=aa))
        ooooo += aa
    midi.save(output_midi_file)


if __name__ == "__main__":
    a = Image.open("assets/badapple/a.jpg")
    a = a.resize((round(a.width * 44 / a.height), 44))
    # a = a.resize((127, round(a.height * 127 / a.width)))
    a.save("assets/badapple/a2.jpg")
    pil_to_midi(a, "assets/test.mid")
