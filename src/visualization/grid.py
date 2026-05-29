import cv2

from src.config import CROP_LEFT, CROP_RIGHT, CROP_TOP_1920, CROP_TOP_OTHER


def _processing_area(frame):
    h, w = frame.shape[:2]
    left = int(w * CROP_LEFT)
    right = int(w * (1 - CROP_RIGHT))
    top = int(h * (CROP_TOP_1920 if w == 1920 else CROP_TOP_OTHER))
    return left, top, right, h


def draw_processing_grid(frame):
    overlay = frame.copy()
    left, top, right, bottom = _processing_area(frame)
    roi_w = right - left
    roi_h = bottom - top
    if roi_w <= 0 or roi_h <= 0:
        return frame

    color = (190, 215, 220)
    accent = (245, 245, 245)
    cell_w = max(110, roi_w // 8)
    cell_h = max(90, roi_h // 6)

    for x in range(left + cell_w, right, cell_w):
        cv2.line(overlay, (x, top), (x, bottom), color, 1, cv2.LINE_AA)
    for y in range(top + cell_h, bottom, cell_h):
        cv2.line(overlay, (left, y), (right, y), color, 1, cv2.LINE_AA)

    cv2.rectangle(overlay, (left, top), (right - 1, bottom - 1), accent, 2, cv2.LINE_AA)
    corner = min(54, max(24, roi_w // 18))
    for sx in (left, right - 1):
        dx = corner if sx == left else -corner
        cv2.line(overlay, (sx, top), (sx + dx, top), accent, 3, cv2.LINE_AA)
        cv2.line(overlay, (sx, bottom - 1), (sx + dx, bottom - 1), accent, 3, cv2.LINE_AA)
    for sy in (top, bottom - 1):
        dy = corner if sy == top else -corner
        cv2.line(overlay, (left, sy), (left, sy + dy), accent, 3, cv2.LINE_AA)
        cv2.line(overlay, (right - 1, sy), (right - 1, sy + dy), accent, 3, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)
