import onnxruntime as ort
from src.backends.base import InferenceBackend

class ONNXBackend(InferenceBackend):

    def __init__(self, num_threads=4):
        self._session = None
        self._input_name = ''
        self._num_threads = num_threads

    def load_model(self, path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._num_threads
        opts.inter_op_num_threads = 1

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._session = ort.InferenceSession(path, sess_options=opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def infer(self, tensor):
        assert self._session is not None, 'model not loaded'
        return self._session.run(None, {self._input_name: tensor})[0]
