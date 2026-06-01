import os

class InferenceBackend:

    def load_model(self, path):
        raise NotImplementedError

    def infer(self, tensor):
        raise NotImplementedError
