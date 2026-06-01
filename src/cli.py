import argparse
import os

from src.config import Config
from src.controller import ProcessingController


class CLIInterface:

    def __init__(self):
        self._controller = ProcessingController()

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
        result = self._controller.run(config)
        print(result)