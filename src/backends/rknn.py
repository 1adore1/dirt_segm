from src.backends.base import InferenceBackend


class RKNNBackend(InferenceBackend):

    def __init__(self, core_mask=None):
        self._rknn = None
        self._core_mask = core_mask

    def load_model(self, path):
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            raise RuntimeError('rknnlite is not available on this platform')

        if self._core_mask is None:
            self._core_mask = RKNNLite.NPU_CORE_AUTO

        self._rknn = RKNNLite()

        ret = self._rknn.load_rknn(path)
        if ret != 0:
            raise RuntimeError(f'failed to load rknn model: {path} (code {ret})')

        ret = self._rknn.init_runtime(core_mask=self._core_mask)
        if ret != 0:
            raise RuntimeError(f'failed to init rknn runtime (code {ret})')
        
    def infer(self, tensor):
        if self._rknn is None:
            raise RuntimeError('model not loaded, call load_model() first')
        return self._rknn.inference(inputs=[tensor])[0]

    def release(self):
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def __del__(self):
        self.release()
