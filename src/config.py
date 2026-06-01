import numpy as np


CLASS_NAMES = [
    'Curb',
    'Wet mud',
    'Wet road',
    'Road marking',
    'Trash',
    'Road feature',
    'Dry mud',
]

BACKEND_BY_EXT = {
    '.onnx': 'onnx',
    '.rknn': 'rknn',
}

CLASS_COLORS = {
    0: (250, 250, 55),
    1: (50, 100, 250), # (51, 221, 255),
    2: (150, 150, 200),
    3: (115, 51, 128),
    4: (250, 50, 83),
    5: (255, 153, 26),
    6: (2, 164, 138),
}

DIRTY_CLASSES = {
    1: 2.0,
    4: 1.5,
    6: 1.0,
}

CISS_MAX = 2.0

CISS_THRESHOLDS = {
    'clean': 0.03,
    'moderate': 0.10,
}

INPUT_SIZE = (800, 800)

CROP_LEFT = 0.17
CROP_RIGHT = 0.17
CROP_TOP_1920 = 0.55
CROP_TOP_OTHER = 0.35

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.MP4', '.MKV'}

GRAPH_WINDOW_SEC = 30
SMOOTHING_WINDOW = 5

class Config:

    def __init__(self, model_path, input_path, output_dir, threshold=0.5, alpha=0.45, fps=25.0):
        self.model_path = model_path
        self.input_path = input_path
        self.output_dir = output_dir
        self.threshold = threshold
        self.alpha = alpha
        self.fps = fps
