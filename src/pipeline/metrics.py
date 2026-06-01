import collections

import numpy as np

from src.config import DIRTY_CLASSES, GRAPH_WINDOW_SEC, SMOOTHING_WINDOW

class MetricsCalculator:

    def __init__(self, fps=25.0, window_sec=GRAPH_WINDOW_SEC):
        self._fps = fps
        self._window_sec = window_sec
        self._scores = []
        self._t = 0.0
        max_pts = int(window_sec * fps) + SMOOTHING_WINDOW + 5
        self.time_axis = collections.deque(maxlen=max_pts)
        self.class_history = {cls_id: collections.deque(maxlen=max_pts)
                              for cls_id in DIRTY_CLASSES}

    def add_frame(self, ciss, class_scores):
        self._scores.append(ciss)
        self.time_axis.append(self._t)
        for cls_id in DIRTY_CLASSES:
            self.class_history[cls_id].append(class_scores.get(cls_id, 0.0))
        self._t += 1.0 / self._fps

    def calc_total(self):
        if not self._scores:
            return 0.0
        return float(np.mean(self._scores))

    def reset(self, fps=None):
        if fps is not None:
            self._fps = fps
        self._scores.clear()
        self._t = 0.0
        self.time_axis.clear()
        for q in self.class_history.values():
            q.clear()

