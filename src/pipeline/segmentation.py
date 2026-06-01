from src.pipeline.preprocessor import Preprocessor
from src.pipeline.postprocessor import Postprocessor


class SegmentationPipeline:

    def __init__(self, backend, threshold=0.65, alpha=0.45):
        self._backend = backend
        self._preprocessor = Preprocessor()
        self._postprocessor = Postprocessor(threshold=threshold, alpha=alpha)

    def process_frame(self, bgr_frame):
        orig, cropped, tensor, scale, pad_top, pad_left = self._preprocessor(bgr_frame)
        logits = self._backend.infer(tensor)
        return self._postprocessor(logits, orig, cropped, scale, pad_top, pad_left)
