import cv2
import numpy as np

from src.config import CLASS_COLORS, CROP_LEFT, CROP_RIGHT, CROP_TOP_1920, CROP_TOP_OTHER, DIRTY_CLASSES
from src.entities import SegmentationResult

class Postprocessor:

    def __init__(self, threshold=0.5, alpha=0.45, colors=None):
        self.threshold = threshold
        self.alpha = alpha
        self.colors = colors or CLASS_COLORS

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def build_mask(self, logits, orig_hw, scale, pad_top, pad_left):
        probs = self._sigmoid(logits[0])

        for c in range(probs.shape[0]):
            probs[c] = cv2.GaussianBlur(probs[c], (5, 5), sigmaX=1.0)

        masks_bin = (probs > self.threshold).astype(np.uint8)

        nh = int(round(orig_hw[0] * scale))
        nw = int(round(orig_hw[1] * scale))
        masks_crop = masks_bin[:, pad_top:pad_top + nh, pad_left:pad_left + nw]

        oh, ow = orig_hw
        masks_big = np.zeros((masks_bin.shape[0], oh, ow), dtype=np.uint8)
        for c in range(masks_bin.shape[0]):
            masks_big[c] = cv2.resize(masks_crop[c], (ow, oh), interpolation=cv2.INTER_NEAREST)
        return masks_big

    def overlay_mask(self, bgr_image, masks):
        color_layer = np.zeros_like(bgr_image, dtype=np.float32)
        count_layer = np.zeros(bgr_image.shape[:2], dtype=np.float32)

        for cls_id, color_rgb in self.colors.items():
            if cls_id >= masks.shape[0] or masks[cls_id].max() == 0:
                continue
            where = masks[cls_id] == 1
            color_bgr = np.array(color_rgb[::-1], dtype=np.float32)
            color_layer[where] += color_bgr
            count_layer[where] += 1.0

        active = count_layer > 0
        color_layer[active] /= count_layer[active, np.newaxis]
        color_uint8 = np.clip(color_layer, 0, 255).astype(np.uint8)

        result = bgr_image.copy()
        result[active] = cv2.addWeighted(
            bgr_image, 1.0 - self.alpha,
            color_uint8, self.alpha, 0
        )[active]
        return result

    def calc_ciss(self, masks):
        total = masks.shape[1] * masks.shape[2]
        if total == 0:
            return 0.0, {}
        ciss = 0.0
        class_scores = {}
        for cls_id, weight in DIRTY_CLASSES.items():
            if cls_id < masks.shape[0]:
                score = float(masks[cls_id].sum()) / total
                class_scores[cls_id] = score
                ciss += weight * score
        return ciss, class_scores

    def __call__(self, logits, orig_bgr, cropped_bgr, scale, pad_top, pad_left):
        masks = self.build_mask(logits, cropped_bgr.shape[:2], scale, pad_top, pad_left)
        overlay_crop = self.overlay_mask(cropped_bgr, masks)
        ciss, class_scores = self.calc_ciss(masks)

        h, w = orig_bgr.shape[:2]
        left = int(w * CROP_LEFT)
        right = int(w * (1 - CROP_RIGHT))
        top = int(h * (CROP_TOP_1920 if w == 1920 else CROP_TOP_OTHER))
        result_frame = orig_bgr.copy()
        result_frame[top:, left:right] = overlay_crop

        return SegmentationResult(overlay_image=result_frame, ciss=ciss, class_scores=class_scores)
