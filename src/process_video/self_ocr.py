import math
import subprocess
from decimal import Decimal
from pathlib import Path

import cv2
import easyocr
import ffmpy
import numpy as np
from cv2.typing import MatLike

files = [
    r"E:\Taiki\bapple\RPReplay_Final1738576908.mp4",
    r"E:\Taiki\bapple\RPReplay_Final1738577625.mp4",
    r"E:\Taiki\bapple\RPReplay_Final1738578181.mp4",
    r"E:\Taiki\bapple\RPReplay_Final1738578794.mp4",
    r"E:\Taiki\bapple\RPReplay_Final1738579593.mp4",
    r"E:\Taiki\bapple\RPReplay_Final1738581900.mp4",
]
p_dir = p_dir = Path(r"E:\Taiki\bapple\pframes")

missing = """
# 5
[2423](2388-2499)
"""


def get_frame(video_path, idx: int) -> MatLike | list[MatLike]:
    cmd = ffmpy.FFmpeg(
        inputs={video_path: None},
        outputs={
            "pipe:1": rf"-vf select=eq(n\\,{idx}) -vframes 1 -f image2pipe -c:v png -y -pix_fmt bgr24"
        },
    )
    stdout, _ = cmd.run(stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    if not isinstance(stdout, bytes):
        raise ValueError("Could not get image")
    img = cv2.imdecode(np.frombuffer(stdout, np.uint8), 0)
    return img


# https://chatgpt.com/share/6816de74-8134-8000-aab4-998d4286d823
def get_frames(video_path, idx: int, idx2: int):
    select_filter = f"select='between(n\\,{idx}\\,{idx2 - 1})',setpts=N/FRAME_RATE/TB"

    cmd = ffmpy.FFmpeg(
        inputs={video_path: None},
        outputs={"pipe:1": f"-vf {select_filter} -vsync 0 -f image2pipe -c:v png -pix_fmt rgb24"},
    )
    stdout, _ = cmd.run(stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    if not isinstance(stdout, bytes):
        raise ValueError("Could not get images")

    # 複数の PNG が連結されたバイナリを OpenCV で分割・デコード
    images = []
    data = stdout
    while data:
        idx = data.find(b"\x89PNG\r\n\x1a\n")  # PNG ヘッダー
        if idx == -1:
            break
        end = data.find(b"IEND\xaeB`\x82", idx)
        if end == -1:
            break
        end += 8  # IEND チャンクの終端まで含む
        img_data = data[idx:end]
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
        data = data[end:]

    return images


def cut_image(src: MatLike, p0: tuple[int, int], p1: tuple[int, int]) -> MatLike:
    return src[p0[1] : p1[1], p0[0] : p1[0]].copy()


def beat2tick(beat: float):
    dec = Decimal(str(beat)) - 1
    if beat.is_integer():
        index = 4 * dec
    else:
        index = 4 * math.floor(dec) + (dec % 1 * 10) - 1
    return int(index) * 480


positions: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "ruler": ((180, 143), (1874, 200)),
    "ruler_time": ((180, 148), (1874, 171)),
    "ruler_separators": ((180, 185), (1874, 193)),
    "ruler_rectangle": ((848, 1), (894, 28)),
    "sbar": ((0, 0), (1920, 30)),
}
heights: dict[str, tuple[int, int]] = {"frame": (284, 1340)}
missing_fs = [[] for _ in range(6)]
video_no = 0
for i in missing.split("\n"):
    if i == "":
        continue
    elif i[0] == "#":
        video_no = int(i[2]) - 1
    else:
        missing_fs[video_no].append(
            [
                eval(i[: i.index("]") + 1]),
                [int(i[i.index("(") + 1 : i.index("-")]), int(i[i.index("-") + 1 : -1])],
            ]
        )

thresh = 185
ocr = easyocr.Reader(["en"], quantize=True)
diameter = 3

beat = 0.0
for f, i in zip(files, missing_fs):
    for tgs, candidate in i:
        for tg in tgs:
            img = get_frames(f, candidate[0] - 2, candidate[1])
            for original in img:
                grayscaled = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                _, threshed = cv2.threshold(
                    cv2.bitwise_not(grayscaled), thresh, 255, cv2.THRESH_BINARY
                )
                # kernel = np.ones((2, 2), np.uint8)
                # img = cv2.erode(img, kernel, iterations=1)
                # 区切り棒などを消す
                ruler_img = cut_image(threshed, *positions["ruler_time"])
                h, w = ruler_img.shape
                ruler_img = cv2.rectangle(ruler_img, *positions["ruler_rectangle"], 255, 3)
                ruler_img[:, np.count_nonzero(ruler_img == 0, axis=0) > h - 5] = 255

                expanded = cv2.resize(
                    ruler_img,
                    None,
                    fx=diameter,
                    fy=diameter,
                    interpolation=cv2.INTER_NEAREST_EXACT,
                )
                _, expanded = cv2.threshold(expanded, thresh, 255, cv2.THRESH_BINARY)
                detected = ocr.detect(
                    expanded,
                    text_threshold=0.5,
                    canvas_size=expanded.shape[1],
                )
                if not detected:
                    print("検出されなかった(OCR)")
                    continue

                start = detected[0][0][0][0], detected[0][0][0][2]
                end = detected[0][0][0][1], detected[0][0][0][3]
                start = list(map(lambda p: p // diameter, start))
                end = list(map(lambda p: p // diameter, end))

                debug_img = cv2.rectangle(
                    original,
                    (
                        start[0] + positions["ruler_time"][0][0],
                        start[1] + positions["ruler_time"][0][1],
                    ),
                    (
                        end[0] + positions["ruler_time"][0][0],
                        end[1] + positions["ruler_time"][0][1],
                    ),
                    (0, 0, 255),
                    1,
                )
                cv2.imshow("win", debug_img)
                cv2.setWindowTitle("win", f"beat: {beat}")
                key = cv2.waitKeyEx(0)
                if key == 115:  # s - skip
                    break
                if key == 99:  # c - continue
                    continue
                if key == 2555904:  # right arrow
                    start = detected[0][0][1][0], detected[0][0][1][2]
                    end = detected[0][0][1][1], detected[0][0][1][3]
                    start = list(map(lambda p: p // diameter, start))
                    end = list(map(lambda p: p // diameter, end))

                    debug_img = cv2.rectangle(
                        original,
                        (
                            start[0] + positions["ruler_time"][0][0],
                            start[1] + positions["ruler_time"][0][1],
                        ),
                        (
                            end[0] + positions["ruler_time"][0][0],
                            end[1] + positions["ruler_time"][0][1],
                        ),
                        (0, 0, 255),
                        1,
                    )
                    cv2.imshow("win", debug_img)
                    key = cv2.waitKeyEx(0)
                if key == 108:  # l - change beat
                    beat = float(input("beatを入力してねー>> "))

                detected_time = beat2tick(beat)

                sep_img = cut_image(threshed, *positions["ruler_separators"])
                sep_candidates = np.where(
                    np.count_nonzero(sep_img == 0, axis=0) > sep_img.shape[0] - 2
                )[0]
                sep_points = sep_candidates[np.ediff1d(sep_candidates, to_end=[0]) != 1]
                nearest_idx = np.abs(np.array(sep_points) - start[0]).argmin()

                seps = [
                    (
                        n,
                        detected_time + 30 * (idx - nearest_idx),
                        idx - nearest_idx,
                    )  # (x座標, tick, 基準からn本目)
                    for idx, n in enumerate(sep_points)
                ]
                min_t = seps[0][1]
                max_t = seps[-1][1]
                nearly_frames = math.ceil((min_t - 120) / 660) + 1
                tg_start = 660 * (nearly_frames - 1) + 120
                tg_end = tg_start + 540
                if tg_start >= min_t and max_t >= tg_end:
                    print(f"フレーム{nearly_frames}が正確にうつっています")
                else:
                    print("映っていないのでcontinue")
                    continue

                fseps = [j for j in seps if j[1] == tg_start or j[1] == tg_end]
                if len(fseps) != 2:
                    print("前代未聞のトラブル発生, 不明なerror")

                frame = cut_image(
                    original,
                    (fseps[0][0] + positions["ruler_time"][0][0], heights["frame"][0]),
                    (fseps[1][0] + positions["ruler_time"][0][0], heights["frame"][1]),
                )
                _, threshed2 = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
                seek_x = np.where(
                    np.count_nonzero(threshed2 == 255, axis=0) > threshed2.shape[0] - 2
                )[0][0]
                frame[:, seek_x - 3 : seek_x + 4] = np.mean(frame[:, seek_x - 8 : seek_x - 3], axis=1, keepdims=True)  # type: ignore
                filename = str(nearly_frames).zfill(6) + ".png"
                cv2.imwrite(str(p_dir.joinpath(filename)), frame)
                cv2.imwrite(
                    str(p_dir.joinpath("ruler/" + filename)),
                    cut_image(original, *positions["ruler"]),
                )
                cv2.imwrite(
                    str(p_dir.joinpath("sbar/" + filename)),
                    cut_image(original, *positions["sbar"]),
                )
            print("FFmpegging")
