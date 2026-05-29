from src.config import BACKEND_BY_EXT, Config, VIDEO_EXTENSIONS
from src.pipeline.preprocessor import Preprocessor
from src.pipeline.postprocessor import Postprocessor
from src.pipeline.metrics import MetricsCalculator
from src.visualization.renderer import OverlayRenderer
from src.io import FileManager
from src.backends.onnx import ONNXBackend
from src.backends.rknn import RKNNBackend

_BACKENDS = {
    'onnx': ONNXBackend,
    'rknn': RKNNBackend,
}

def _backend_from_path(model_path):
    ext = os.path.splitext(model_path)[1].lower()
    name = BACKEND_BY_EXT.get(ext)
    if name is None:
        raise ValueError(f'unsupported model format: {ext}, expected .onnx or .rknn')
    return _BACKENDS[name]()

import argparse
import os
import cv2
from tqdm import tqdm

class CLIInterface:

    def __init__(self):
        self._backend = None
        self._preprocessor = Preprocessor()
        self._postprocessor = None
        self._metrics = MetricsCalculator()
        self._overlay = OverlayRenderer()
        self._file_manager = FileManager()

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', required=True)
        parser.add_argument('--input', required=True)
        parser.add_argument('--output', required=True)
        parser.add_argument('--threshold', type=float, default=0.5)
        parser.add_argument('--alpha', type=float, default=0.45)
        parser.add_argument('--fps', type=float, default=25.0)
        ns = parser.parse_args(args)
        return Config(
            model_path=ns.model,
            input_path=ns.input,
            output_dir=ns.output,
            threshold=ns.threshold,
            alpha=ns.alpha,
            fps=ns.fps,
        )

    def run(self, config):
        self._backend = _backend_from_path(config.model_path)
        self._postprocessor = Postprocessor(threshold=config.threshold, alpha=config.alpha)
        self._backend.load_model(config.model_path)
        self._file_manager.ensure_dir(config.output_dir)

        ext = os.path.splitext(config.input_path)[1]
        if ext in VIDEO_EXTENSIONS:
            self._process_video(config)
        else:
            self._process_image(config)

    def _run_single_frame(self, bgr_frame):
        orig, cropped, tensor, scale, pad_top, pad_left = self._preprocessor(bgr_frame)
        logits = self._backend.infer(tensor)
        return self._postprocessor(logits, orig, cropped, scale, pad_top, pad_left)

    def _process_image(self, config):
        frame = self._file_manager.read_image(config.input_path)
        self._metrics.reset(fps=config.fps)
        result = self._run_single_frame(frame)
        self._metrics.add_frame(result.ciss, result.class_scores)
        out_frame = self._overlay.render(result.overlay_image, result, self._metrics)

        name = os.path.splitext(os.path.basename(config.input_path))[0]
        out_path = os.path.join(config.output_dir, f'{name}_segm.jpg')
        self._file_manager.save_image(out_path, out_frame)
        print(f'CISS: {result.ciss * 100:.1f}%')

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
            result = self._run_single_frame(frame)
            self._metrics.add_frame(result.ciss, result.class_scores)
            out_frame = self._overlay.render(result.overlay_image, result, self._metrics)

            if writer is None:
                h, w = out_frame.shape[:2]
                writer = self._file_manager.open_video_writer(out_path, fps, (w, h))
            writer.write(out_frame)

        cap.release()
        if writer is not None:
            writer.release()

        print(f'Total CISS: {self._metrics.calc_total() * 100:.1f}%')

