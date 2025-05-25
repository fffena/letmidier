import math
from pathlib import PurePath
from typing import Literal

import cv2
import mido
import numpy as np
import tqdm
from mido import Message, MidiFile
from PIL import Image


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
    info.sort(key=lambda x: x[2])
    print(info)
    base_time = 0
    for opcode, note, t in info:
        time = t * 15 - base_time
        msg_type = "note_on" if opcode == 0 else "note_off"
        track.append(Message(msg_type, note=note, velocity=100, time=time))
        base_time += time
    midi.save(output_midi_file)


def video_to_midi(video_fp: str | PurePath, output_fp: str | PurePath, a):
    if isinstance(video_fp, PurePath):
        video_fp = str(video_fp)
    cap = cv2.VideoCapture(video_fp)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file cannot be found: {video_fp}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pg_bar = tqdm.tqdm(total=total_frames)

    midi = MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120)))

    space = 120
    pitch_offset = 120
    frame_length = 540 + space
    now_t = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # if cap.get(cv2.CAP_PROP_POS_FRAMES) % 2:
        #     continue

        # サイズ調整などの加工
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, dsize=(round(w * (44 / h)), 44))
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, img_array = cv2.threshold(grayscaled, 0, 255, cv2.THRESH_OTSU)

        counted = count_same(img_array)

        info: list[tuple[Literal[0, 1], int, int]] = []
        for height_offset, i in enumerate(counted):
            now = 0
            for im_data in i:
                if im_data[0] == 0:
                    o = pitch_offset - height_offset
                    info.append((0, o, now))
                    info.append((1, o, now + im_data[1]))
                now += im_data[1]
        info.sort(key=lambda x: x[2])

        base_t = 0
        for ind, (opcode, note, t) in enumerate(info):
            time = math.floor(t * a - base_t)
            shift = frame_length - (now_t % frame_length)
            if ind == 0:
                time += space
                base_t -= space

                # ズレの修正
                if shift != frame_length:
                    time += shift
                    base_t -= shift
            msg_type = "note_on" if opcode == 0 else "note_off"
            track.append(Message(msg_type, note=note, velocity=100, time=time))

            base_t += time
            now_t += time

        pg_bar.n = frame_count
        pg_bar.refresh()
    pg_bar.close()
    midi.save(output_fp)
