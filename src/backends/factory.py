import os
from src.config import BACKEND_BY_EXT

def load_backend(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'model not found: {model_path}')
    ext = os.path.splitext(model_path)[1].lower()
    backend_name = BACKEND_BY_EXT.get(ext)
    if backend_name is None:
        raise ValueError(f'unsupported model format: {ext}')

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
