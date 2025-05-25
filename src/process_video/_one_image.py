import math
from decimal import Decimal
from enum import Enum

import cv2
import easyocr
import numpy as np
from cv2.typing import MatLike


eprint = lambda x: exit(print(x))

positions: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "ruler": ((180, 143), (1874, 200)),
    "ruler_time": ((180, 148), (1874, 171)),
    "ruler_separators": ((180, 185), (1874, 193)),
    "ruler_rectangle": ((848, 1), (894, 28)),
    "sbar": ((0, 0), (1920, 30)),
}
heights: dict[str, tuple[int, int]] = {"frame": (284, 1340)}


class DebugImageType(Enum):
    NONE = 0
    FULL = 1
    RULER = 2
    RULER_TIME = 3
    FRAME = 4


def cut_image(src: MatLike, p0: tuple[int, int], p1: tuple[int, int]) -> MatLike:
    return src[p0[1] : p1[1], p0[0] : p1[0]].copy()


def beat2tick(beat: float):
    dec = Decimal(str(beat)) - 1
    if beat.is_integer():
        index = 4 * dec
    else:
        index = 4 * math.floor(dec) + (dec % 1 * 10) - 1
    return int(index) * 480


ocr = easyocr.Reader(["en"], quantize=True)
thresh = 185
diameter = 3  # ocrに投げるときの拡大倍率

debug: DebugImageType = DebugImageType.FULL
if debug != DebugImageType.NONE:
    cv2.namedWindow("win", cv2.WINDOW_NORMAL)

original = cv2.imread(r"E:\Taiki\bapple\pframes\orig\004634-2070.png")
grayscaled = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
_, threshed = cv2.threshold(cv2.bitwise_not(grayscaled), thresh, 255, cv2.THRESH_BINARY)
# kernel = np.ones((2, 2), np.uint8)
# img = cv2.erode(img, kernel, iterations=1)
# 区切り棒などを消す
ruler_img = cut_image(threshed, *positions["ruler_time"])
h, w = ruler_img.shape
ruler_img = cv2.rectangle(ruler_img, *positions["ruler_rectangle"], 255, 3)
ruler_img[:, np.count_nonzero(ruler_img == 0, axis=0) > h - 5] = 255

# 一度拡大してOCRに投げたほうが精度が良くなった
# expanded = cv2.GaussianBlur(ruler_img, (3, 3), 0)
expanded = cv2.resize(
    ruler_img, None, fx=diameter, fy=diameter, interpolation=cv2.INTER_NEAREST_EXACT
)
_, expanded = cv2.threshold(expanded, thresh, 255, cv2.THRESH_BINARY)
detected: list[tuple[list[list[int]], str, float]] = ocr.readtext(
    expanded,
    text_threshold=0.5,
    canvas_size=expanded.shape[1],
    allowlist="0123456789.",
    batch_size=3,
)  # type: ignore
if not detected:
    eprint("検出されなかった(OCR)")

sep_img = cut_image(threshed, *positions["ruler_separators"])
sep_candidates = np.where(np.count_nonzero(sep_img == 0, axis=0) > sep_img.shape[0] - 2)[0]
sep_points = sep_candidates[np.ediff1d(sep_candidates, to_end=[0]) != 1]

for j in detected:
    points = j[0]
    text = j[1]
    reliability = j[2]
    try:
        detected_time = math.floor(float(text.rstrip(".")) * 10) / 10
    except ValueError:
        continue
    if str(detected_time)[-1] not in ["0", "2", "3", "4"]:
        continue
    if sep_points[0] > points[0][0]:
        continue
    start = points[0]
    end = points[2]
    break
else:
    exit(eprint("値がすべての条件に一致しなかった"))

print("検出された値: ", detected_time)
detected_time = beat2tick(detected_time)

start = list(map(lambda p: p // diameter, start))
end = list(map(lambda p: p // diameter, end))

nearest_idx = np.abs(np.array(sep_points) - start[0]).argmin()

# マジックナンバー多用ゾーン
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
# 終わり

fseps = [j for j in seps if j[1] == tg_start or j[1] == tg_end]
if len(fseps) != 2:
    eprint("前代未聞のトラブル発生, 不明なerror")

frame = cut_image(
    original,
    (fseps[0][0] + positions["ruler_time"][0][0], heights["frame"][0]),
    (fseps[1][0] + positions["ruler_time"][0][0], heights["frame"][1]),
)
_, threshed2 = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
seek_x = np.where(np.count_nonzero(threshed2 == 255, axis=0) > threshed2.shape[0] - 2)[0][0]
# なんで型エラー出るかわからん
# 再生位置の白いバーを付近の色の平均値で上書き
frame[:, seek_x - 3 : seek_x + 4] = np.mean(frame[:, seek_x - 8 : seek_x - 3], axis=1, keepdims=True)  # type: ignore

if debug == DebugImageType.FULL:
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
    debug_img = cv2.rectangle(
        debug_img,
        (fseps[0][0] + positions["ruler_time"][0][0], heights["frame"][0]),
        (fseps[1][0] + positions["ruler_time"][0][0], heights["frame"][1]),
        (0, 0, 255),
        2,
    )
elif debug == DebugImageType.RULER:
    debug_img = cv2.cvtColor(cut_image(threshed, *positions["ruler"]), cv2.COLOR_GRAY2BGR)
    debug_img = cv2.rectangle(
        debug_img,
        (start[0], start[1] + positions["ruler_time"][0][1]),
        (end[0], end[1] + positions["ruler_time"][0][1]),
        (0, 0, 255),
        1,
    )
elif debug == DebugImageType.RULER_TIME:
    debug_img = cv2.cvtColor(ruler_img, cv2.COLOR_GRAY2BGR)
    debug_img = cv2.rectangle(debug_img, start, end, (0, 0, 255), 1)
elif debug == DebugImageType.FRAME:
    debug_img = frame

# 区切り線の位置の描画
if debug == DebugImageType.RULER or debug == DebugImageType.RULER_TIME:
    debug_img = cv2.rectangle(debug_img, (seps[0][0], 0), (seps[0][0], 100), (255, 0, 0), 2)
    debug_img = cv2.rectangle(debug_img, (seps[-1][0], 0), (seps[-1][0], 100), (255, 0, 0), 2)
    nearest_x = seps[nearest_idx][0]
    debug_img = cv2.rectangle(debug_img, (nearest_x, 0), (nearest_x, 100), (255, 0, 0), 2)
elif debug == DebugImageType.FULL:
    for func in (min, max):
        debug_img = cv2.rectangle(
            debug_img,
            (
                func(seps)[0] + positions["ruler_time"][0][0],
                positions["ruler_time"][0][1],
            ),
            (
                func(seps)[0] + positions["ruler_time"][0][0],
                90 + positions["ruler_time"][0][1],
            ),
            (255, 0, 0),
            2,
        )
if debug != DebugImageType.NONE:
    cv2.imshow("win", debug_img)
    cv2.setWindowTitle("win", f"{str(nearly_frames).zfill(6)}.png")
    cv2.waitKey(0)
