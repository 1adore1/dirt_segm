import os

import cv2
from tqdm import tqdm

from src.config import VIDEO_EXTENSIONS
from src.pipeline.metrics import MetricsCalculator
from src.pipeline.segmentation import SegmentationPipeline
from src.visualization.renderer import OverlayRenderer
from src.io import FileManager

from src.backends.factory import load_backend

class ProcessingController:

    def __init__(self):
        self._metrics = MetricsCalculator()
        self._overlay = OverlayRenderer()
        self._file_manager = FileManager()

    def run(self, config):
        backend = load_backend(config.model_path)
        self._pipeline = SegmentationPipeline(backend, threshold=config.threshold, alpha=config.alpha)
        self._file_manager.ensure_dir(config.output_dir)

        ext = os.path.splitext(config.input_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return self._process_video(config)
        return self._process_image(config)

    def _process_image(self, config):
        frame = self._file_manager.read_image(config.input_path)
        self._metrics.reset(fps=config.fps)
        result = self._pipeline.process_frame(frame)
        self._metrics.add_frame(result.ciss, result.class_scores)
        out_frame = self._overlay.render(result.overlay_image, result, self._metrics)

        name = os.path.splitext(os.path.basename(config.input_path))[0]
        out_path = os.path.join(config.output_dir, f'{name}_segm.jpg')
        self._file_manager.save_image(out_path, out_frame)
        return f'CISS: {result.ciss * 100:.1f}%'

    def _process_video(self, config):
        cap = self._file_manager.read_video(config.input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or config.fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._metrics.reset(fps=fps)
        self._overlay.reset()

        name = os.path.splitext(os.path.basename(config.input_path))[0]
        out_path = os.path.join(config.output_dir, f'{name}_segm.mp4')
        writer = None

        for _ in tqdm(range(total), desc='Processing'):
            ret, frame = cap.read()
            if not ret:
                break
            result = self._pipeline.process_frame(frame)
            self._metrics.add_frame(result.ciss, result.class_scores)
            out_frame = self._overlay.render(result.overlay_image, result, self._metrics)

            if writer is None:
                h, w = out_frame.shape[:2]
                writer = self._file_manager.open_video_writer(out_path, fps, (w, h))
            writer.write(out_frame)

        cap.release()
        if writer is not None:
            writer.release()

        return f'Total CISS: {self._metrics.calc_total() * 100:.1f}%'
