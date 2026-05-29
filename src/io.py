import os
import cv2

class FileManager:

    def ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)

    def read_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f'image not found: {path}')
        return img

    def read_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f'video not found or corrupted: {path}')
        return cap

    def save_image(self, path, image):
        self.ensure_dir(os.path.dirname(path) or '.')
        cv2.imwrite(path, image)

    def open_video_writer(self, path, fps, frame_size):
        self.ensure_dir(os.path.dirname(path) or '.')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(path, fourcc, fps, frame_size)