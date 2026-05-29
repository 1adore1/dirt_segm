class SegmentationResult:
    def __init__(self, overlay_image, ciss, class_scores):
        self.overlay_image = overlay_image
        self.ciss = ciss
        self.class_scores = class_scores