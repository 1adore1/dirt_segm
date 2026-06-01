import os

from src.config import BACKEND_BY_EXT


class InferenceBackend:

    def load(self, model_path):
        ext = os.path.splitext(model_path)[1].lower()
        backend_name = BACKEND_BY_EXT.get(ext)
        if backend_name is None:
            raise ValueError(f'unsupported model format: {ext}, expected .onnx or .rknn')

        if backend_name == 'onnx':
            from src.backends.onnx import ONNXBackend
            backend = ONNXBackend()
        elif backend_name == 'rknn':
            from src.backends.rknn import RKNNBackend
            backend = RKNNBackend()
        else:
            raise ValueError(f'unsupported backend: {backend_name}')

        backend.load_model(model_path)
        return backend

    def load_model(self, path):
        raise NotImplementedError

    def infer(self, tensor):
        raise NotImplementedError
