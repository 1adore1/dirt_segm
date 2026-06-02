import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import CLASS_COLORS, CLASS_NAMES, CISS_THRESHOLDS, DIRTY_CLASSES, GRAPH_WINDOW_SEC
from src.pipeline.metrics import MetricsCalculator
from src.visualization.grid import draw_processing_grid


DISPLAY_CLASS_NAMES = {
    1: 'Влажная грязь',
    4: 'Мусор',
    6: 'Сухая грязь',
}

_FONT_CANDIDATES = [
    '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    'C:/Windows/Fonts/segoeui.ttf',
    'C:/Windows/Fonts/arial.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/usr/share/fonts/TTF/DejaVuSans.ttf',
]

def _pil_font(size):
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    # ищем любой доступный TTF через matplotlib
    try:
        from matplotlib import font_manager
        candidates = font_manager.findSystemFonts(fontext='ttf')
        if candidates:
            return ImageFont.truetype(candidates[0], size)
    except Exception:
        pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()




class OverlayRenderer:

    def __init__(self):
        self._fig, self._ax = plt.subplots(figsize=(7.0, 4.0), dpi=110)
        self._canvas = FigureCanvas(self._fig)
        self._turq = (0.0, 1.0, 1.0)
        self._display_ciss = None
        self._display_scores = {cls_id: 0.0 for cls_id in DIRTY_CLASSES}

    def reset(self):
        self._display_ciss = None
        self._display_scores = {cls_id: 0.0 for cls_id in DIRTY_CLASSES}

    def render(self, frame, result, metrics, is_video):
        display_ciss, display_scores = self._smooth_display(
            result.ciss, result.class_scores, metrics._fps
        )
        frame = draw_processing_grid(frame)
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        draw = ImageDraw.Draw(image_pil)
        graph_y = 220
        self._draw_contamination_indicator(draw, image_pil.width, display_ciss, graph_y)
        if is_video:
            graph_img = self._render_line(metrics)
        else:
            graph_img = self._render_bar(metrics)
        if graph_img is not None:
            image_pil.paste(graph_img, (24, graph_y), graph_img)
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGBA2BGR)

    def _smooth_display(self, ciss, class_scores, fps):
        alpha = min(0.22, max(0.08, 3.0 / max(fps, 1.0)))
        if self._display_ciss is None:
            self._display_ciss = ciss
            self._display_scores = {cls_id: class_scores.get(cls_id, 0.0)
                                    for cls_id in DIRTY_CLASSES}
        else:
            self._display_ciss += alpha * (ciss - self._display_ciss)
            for cls_id in DIRTY_CLASSES:
                cur = self._display_scores.get(cls_id, 0.0)
                self._display_scores[cls_id] = cur + alpha * (class_scores.get(cls_id, 0.0) - cur)
        return self._display_ciss, dict(self._display_scores)

    def _render_graph(self, metrics):
        if len(metrics.time_axis) == 0:
            return None
        return self._render_line(metrics)

    def _render_bar(self, metrics):
        ax = self._ax
        ax.clear()

        labels = [DISPLAY_CLASS_NAMES.get(cls_id, CLASS_NAMES[cls_id]) for cls_id in DIRTY_CLASSES]
        values = [metrics.class_history[cls_id][0] * 100 for cls_id in DIRTY_CLASSES]
        colors = [tuple(c / 255.0 for c in CLASS_COLORS[cls_id]) for cls_id in DIRTY_CLASSES]

        bars = ax.barh(labels, values, color=colors, height=0.5)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=14, color=self._turq,
            )

        ax.set_xlim(0, max(10.0, max(values) * 1.3))
        ax.set_xlabel('Площадь, %', fontsize=20, color=self._turq)
        ax.tick_params(colors=self._turq, labelsize=14)

        self._apply_graph_style()
        return self._fig_to_image()

    def _render_line(self, metrics):
        ax = self._ax
        ax.clear()

        t_now = metrics.time_axis[-1]
        idx_range = [i for i, tt in enumerate(metrics.time_axis)
                    if (t_now - tt) <= GRAPH_WINDOW_SEC]
        if not idx_range:
            return None

        max_y = 0.0
        for cls_id in DIRTY_CLASSES:
            x_vals = [metrics.time_axis[i] for i in idx_range]
            y_vals = [metrics.class_history[cls_id][i] * 100 for i in idx_range]
            if not x_vals:
                continue
            color_mpl = tuple(c / 255.0 for c in CLASS_COLORS[cls_id])
            ax.plot(x_vals, y_vals, color=color_mpl, linewidth=3.0,
                    label=DISPLAY_CLASS_NAMES.get(cls_id, CLASS_NAMES[cls_id]))
            if y_vals:
                max_y = max(max_y, max(y_vals))

        ax.set_ylim(0, max(5.0, max_y * 1.2))
        ax.set_xlabel('Время, с', fontsize=20, color=self._turq)
        ax.set_ylabel('Площадь, %', fontsize=20, color=self._turq)

        legend = ax.legend(fontsize=16, loc='upper left')
        for txt in legend.get_texts():
            txt.set_color('white')
        legend.get_frame().set_facecolor((0, 0, 0, 0.3))
        legend.get_frame().set_edgecolor(self._turq)

        self._apply_graph_style()
        return self._fig_to_image()

    def _apply_graph_style(self):
        self._ax.grid(True, color=self._turq, alpha=0.3)
        self._ax.tick_params(colors=self._turq)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(self._turq)
            spine.set_alpha(0.4)
        self._fig.patch.set_alpha(0)
        self._fig.subplots_adjust(left=0.13, right=0.97, bottom=0.20, top=0.95)
        self._ax.patch.set_alpha(0)

    def _fig_to_image(self):
        self._canvas.draw()
        gw, gh = self._fig.canvas.get_width_height()
        rgba = np.frombuffer(self._canvas.buffer_rgba(), dtype=np.uint8).reshape(gh, gw, 4)
        return Image.fromarray(rgba, mode='RGBA')

    def _severity(self, ciss):
        if ciss < CISS_THRESHOLDS['clean']:
            return (55, 225, 145)
        if ciss < CISS_THRESHOLDS['moderate']:
            return (245, 188, 66)
        return (242, 92, 92)

    def _draw_hud_panel(self, draw, xy, radius=16):
        draw.rounded_rectangle(
            xy,
            radius=radius,
            fill=(0, 0, 0, 96),
            outline=(0, 255, 255, 135),
            width=2,
        )

    def _draw_contamination_indicator(self, draw, img_w, ciss, y=220):
        panel_w = 400
        panel_h = 175
        margin = 24
        x = img_w - panel_w - margin
        self._draw_hud_panel(draw, (x, y, x + panel_w, y + panel_h), radius=18)
        color = self._severity(ciss)
        font_title = _pil_font(30)
        font_value = _pil_font(100)
        draw.text((x + 16, y + 14), 'Pollution Index', font=font_title, fill=(0, 255, 255, 220))
        draw.text((x + 24, y + 52), f'{ciss * 100:.1f}%', font=font_value, fill=(*color, 255))        