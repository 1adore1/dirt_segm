import cv2
import numpy as np

from src.config import CROP_LEFT, CROP_RIGHT, CROP_TOP_1920, CROP_TOP_OTHER, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


class Preprocessor:

    def crop(self, image):
        h, w = image.shape[:2]
        left = int(w * CROP_LEFT)
        right = int(w * (1 - CROP_RIGHT))
        top = int(h * (CROP_TOP_1920 if w == 1920 else CROP_TOP_OTHER))
        return image[top:, left:right]

    def letterbox(self, image, size=INPUT_SIZE):
        h_orig, w_orig = image.shape[:2]
        h_t, w_t = size
        scale = min(w_t / w_orig, h_t / h_orig)
        nw = int(round(w_orig * scale))
        nh = int(round(h_orig * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = int(round((h_t - nh) / 2 - 0.1))
        left = int(round((w_t - nw) / 2 - 0.1))
        padded = np.zeros((h_t, w_t, 3), dtype=resized.dtype)
        padded[top:top + nh, left:left + nw] = resized
        return padded, scale, top, left

    def normalize(self, image):
        img = image.astype(np.float32) / 255.0
        return (img - IMAGENET_MEAN) / IMAGENET_STD

    def to_tensor(self, image):
        return np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0).astype(np.float32)

    def __call__(self, bgr_frame):
        cropped = self.crop(bgr_frame)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        lb, scale, pad_top, pad_left = self.letterbox(rgb)
        tensor = self.to_tensor(self.normalize(lb))
        return bgr_frame, cropped, tensor, scale, pad_top, pad_left