from src.config import DIRTY_CLASSES


class CISSCalculator:

    def __init__(self, dirty_classes=None):
        self._dirty_classes = dirty_classes or DIRTY_CLASSES

    def calculate(self, masks):
        total = masks.shape[1] * masks.shape[2]
        if total == 0:
            return 0.0, {}

        ciss = 0.0
        class_scores = {}
        for cls_id, weight in self._dirty_classes.items():
            if cls_id < masks.shape[0]:
                score = float(masks[cls_id].sum()) / total
                class_scores[cls_id] = score
                ciss += weight * score
        return ciss, class_scores
