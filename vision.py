"""
Компьютерное зрение на чистом numpy (без OpenCV — чтобы ставилось на любой Python).

Идея: асфальт трассы — серый (низкая насыщенность цвета), трава/отбойники/небо —
цветные или очень яркие/тёмные. Берём полосу экрана ПЕРЕД машиной, выделяем
"проезжую" серую область, считаем её центр масс по горизонтали и сравниваем
с центром экрана. Получаем ошибку руления в диапазоне [-1, 1].
"""

import numpy as np
import mss


class TrackVision:
    def __init__(self, cfg):
        v = cfg["vision"]
        self.band_top = float(v["band_top"])
        self.band_bottom = float(v["band_bottom"])
        self.sat_max = int(v["sat_max"])
        self.val_min = int(v["val_min"])
        self.val_max = int(v["val_max"])
        self.min_coverage = float(v["min_coverage"])
        self.row_weight = float(v["row_weight"])
        self.adaptive = bool(v.get("adaptive", True))
        self.color_tol = float(v.get("color_tol", 55))
        self.proc_step = max(1, int(v.get("proc_step", 2)))  # прореживание пикселей -> скорость
        self.ref_color = None        # самокалибрующийся цвет асфальта (BGR)
        self.mode = v.get("mode", "racing_line")   # racing_line (цветная линия) | asphalt
        self.line_min = int(v.get("line_min", 95))   # мин. яркость канала для цветной линии
        self.line_diff = int(v.get("line_diff", 28)) # насколько канал должен доминировать

        self.cap = cfg["capture"]
        self._sct = mss.mss()
        self.window_title = self.cap.get("window_title", "Roblox")
        self.using_window = False
        self.refresh_region()

    def _monitor_region(self):
        cap = self.cap
        mon = self._sct.monitors[int(cap.get("monitor", 1))]
        if cap.get("width") and cap.get("height"):
            return {"top": int(cap["top"]), "left": int(cap["left"]),
                    "width": int(cap["width"]), "height": int(cap["height"])}
        return dict(mon)

    def refresh_region(self):
        """
        Если режим 'window' — захватываем ТОЛЬКО окно Roblox (ищем заново,
        вдруг подвинули/свернули). Иначе монитор. Возвращает True, если нашли окно.
        """
        if self.cap.get("mode", "window") == "window":
            import window as _win
            reg = _win.window_region(self.window_title)
            if reg:
                self.window_found_title = reg.pop("title", self.window_title)
                self.region = reg
                self.using_window = True
                return True
        self.region = self._monitor_region()
        self.using_window = False
        return False

    def grab(self) -> np.ndarray:
        """BGRA -> BGR uint8 кадр."""
        raw = np.asarray(self._sct.grab(self.region))
        return raw[:, :, :3]

    def analyze(self, frame: np.ndarray):
        """
        Возвращает (error, confidence):
          error      -1.0 = центр трассы сильно слева, +1.0 = сильно справа.
          confidence  0..1, доля "асфальта" в полосе обзора.
        """
        h, w = frame.shape[:2]
        y0 = int(h * self.band_top)
        y1 = int(h * self.band_bottom)
        band = frame[y0:y1, :, :].astype(np.int16)

        mx = band.max(axis=2)
        mn = band.min(axis=2)
        sat = mx - mn            # насыщенность (грубая, без нормировки) — серый ~ 0
        val = mx                 # яркость

        mask = (sat <= self.sat_max) & (val >= self.val_min) & (val <= self.val_max)

        bh = mask.shape[0]
        # Ряды ближе к машине (внизу полосы) важнее — их вес выше.
        ramp = np.linspace(1.0, self.row_weight, bh).reshape(bh, 1)
        weighted = mask * ramp

        cols = weighted.sum(axis=0)          # вклад каждого столбца
        total = cols.sum()
        max_total = ramp.sum() * w
        confidence = float(total / max_total) if max_total > 0 else 0.0

        if confidence < self.min_coverage:
            return 0.0, confidence

        xs = np.arange(w)
        centroid = float((cols * xs).sum() / total)
        error = (centroid - w / 2.0) / (w / 2.0)
        return float(np.clip(error, -1.0, 1.0)), confidence

    def _segment(self, band: np.ndarray) -> np.ndarray:
        """
        Маска асфальта. Сначала грубая эвристика "серого", затем САМОКАЛИБРОВКА:
        учим реальный цвет трассы по уже найденным пикселям и добираем по нему всё,
        что похоже по цвету (даже если асфальт не идеально серый). Так бот видит
        трассу целиком, как человек, без ручной подгонки порогов.
        """
        mx = band.max(axis=2)
        mn = band.min(axis=2)
        gray = (((mx - mn) <= self.sat_max) & (mx >= self.val_min)
                & (mx <= self.val_max))
        if not self.adaptive:
            return gray.astype(np.float32)
        # учим цвет асфальта по найденным "серым" пикселям
        if gray.sum() > gray.size * 0.02:
            med = np.median(band[gray], axis=0).astype(np.float32)
            self.ref_color = med if self.ref_color is None else 0.9 * self.ref_color + 0.1 * med
        if self.ref_color is not None:
            diff = band.astype(np.float32) - self.ref_color
            dist = np.sqrt((diff * diff).sum(axis=2))
            adaptive = dist < self.color_tol
            return (adaptive | gray).astype(np.float32)
        return gray.astype(np.float32)

    def _far_point(self, mask: np.ndarray) -> float:
        """Гоночная линия: горизонтальная ошибка самой дальней видимой точки трассы."""
        bh, w = mask.shape
        xs = np.arange(w)
        need = w * 0.06
        for r in range(bh):
            block = mask[r:min(r + 4, bh)]
            cs = block.sum(axis=0)
            if cs.sum() > need * block.shape[0]:
                centroid = float((cs * xs).sum() / cs.sum())
                return float(np.clip((centroid - w / 2.0) / (w / 2.0), -1.0, 1.0))
        return 0.0

    def _line_masks(self, band: np.ndarray):
        """Маски цветной гоночной линии: (зелёная, жёлтая, красная)."""
        B = band[:, :, 0]; G = band[:, :, 1]; R = band[:, :, 2]
        m = self.line_min
        d = self.line_diff
        green = (G > m) & (G > R + d) & (G > B + d)
        red = (R > m) & (R > G + d) & (R > B + d)
        yellow = (R > m) & (G > m) & (R > B + d + 15) & (G > B + d + 15)
        return green, yellow, red

    def line_scene(self, frame: np.ndarray):
        """
        Едем по ЦВЕТНОЙ гоночной линии (зелёная/жёлтая/красная) — оптимальная
        траектория, нарисованная в игре. Цвет = подсказка газ/тормоз:
          зелёный -> газ, жёлтый -> сброс, красный -> тормоз (тормозим заранее).
        """
        h, w = frame.shape[:2]
        y0 = int(h * self.band_top)
        y1 = int(h * self.band_bottom)
        s = self.proc_step
        band = frame[y0:y1:s, ::s, :].astype(np.int16)
        green, yellow, red = self._line_masks(band)
        line_mask = (green | yellow | red).astype(np.float32)
        bh = line_mask.shape[0]
        t = max(1, bh // 3)
        far, far_c = self._centroid_err(line_mask[:t])
        mid, mid_c = self._centroid_err(line_mask[t:2 * t])
        near, near_c = self._centroid_err(line_mask[2 * t:])
        conf = (far_c + mid_c + near_c) / 3.0
        line = self._far_point(line_mask)
        curv = far - near

        # подсказка газ/тормоз по цвету ВПЕРЕДИ (верхние 2/3 полосы = даль/средне)
        ahead = slice(0, 2 * t)
        rf = float(red[ahead].sum())
        yf = float(yellow[ahead].sum())
        tot = float(line_mask[ahead].sum()) + 1.0
        if rf / tot > 0.18:
            hint = "brake"
        elif yf / tot > 0.25:
            hint = "coast"
        else:
            hint = "go"
        return {"near": near, "mid": mid, "far": far, "line": line,
                "curv": curv, "conf": conf, "hint": hint}

    def scene(self, frame: np.ndarray):
        """Восприятие дороги. Диспетчер: цветная линия или серый асфальт."""
        if self.mode == "racing_line":
            return self.line_scene(frame)
        h, w = frame.shape[:2]
        y0 = int(h * self.band_top)
        y1 = int(h * self.band_bottom)
        s = self.proc_step
        band = frame[y0:y1:s, ::s, :].astype(np.int16)
        mask = self._segment(band)
        bh = mask.shape[0]
        t = max(1, bh // 3)
        far, far_c = self._centroid_err(mask[:t])
        mid, mid_c = self._centroid_err(mask[t:2 * t])
        near, near_c = self._centroid_err(mask[2 * t:])
        conf = (far_c + mid_c + near_c) / 3.0
        line = self._far_point(mask)
        curv = far - near
        return {"near": near, "mid": mid, "far": far, "line": line,
                "curv": curv, "conf": conf, "hint": None}

    def _centroid_err(self, mask: np.ndarray):
        """Центр масс маски по горизонтали -> (error[-1..1], coverage[0..1])."""
        bh, w = mask.shape
        ramp = np.linspace(1.0, self.row_weight, bh).reshape(bh, 1)
        weighted = mask * ramp
        cols = weighted.sum(axis=0)
        total = cols.sum()
        max_total = ramp.sum() * w
        cov = float(total / max_total) if max_total > 0 else 0.0
        if cov < self.min_coverage * 0.6:
            return 0.0, cov
        xs = np.arange(w)
        centroid = float((cols * xs).sum() / total)
        return float(np.clip((centroid - w / 2.0) / (w / 2.0), -1.0, 1.0)), cov

    def analyze2(self, frame: np.ndarray):
        """
        Возвращает (near_err, far_err, conf):
          near_err — куда уходит трасса прямо перед машиной (для точного руля);
          far_err  — куда трасса уходит ВДАЛИ (для упреждения поворота/торможения);
          conf     — общая доля асфальта в обзоре.
        """
        h, w = frame.shape[:2]
        y0 = int(h * self.band_top)
        y1 = int(h * self.band_bottom)
        band = frame[y0:y1, :, :].astype(np.int16)
        mx = band.max(axis=2)
        mn = band.min(axis=2)
        mask = (((mx - mn) <= self.sat_max) & (mx >= self.val_min)
                & (mx <= self.val_max))
        bh = mask.shape[0]
        mid = bh // 2
        far_err, far_cov = self._centroid_err(mask[:mid])     # верх полосы = даль
        near_err, near_cov = self._centroid_err(mask[mid:])   # низ полосы = близь
        conf = (far_cov + near_cov) / 2.0
        return near_err, far_err, conf

    def racing_line(self, frame: np.ndarray):
        """
        Оптимальная (гоночная) линия без всякой записи: целимся в самую ДАЛЬНЮЮ
        видимую точку трассы. В повороте дальняя достижимая точка лежит на
        внутренней стороне — поэтому машина естественно срезает апекс.
        Возвращает (line_err[-1..1], conf).
        """
        h, w = frame.shape[:2]
        y0 = int(h * self.band_top)
        y1 = int(h * self.band_bottom)
        band = frame[y0:y1, :, :].astype(np.int16)
        mx = band.max(axis=2)
        mn = band.min(axis=2)
        mask = (((mx - mn) <= self.sat_max) & (mx >= self.val_min)
                & (mx <= self.val_max)).astype(np.float32)
        bh, _ = mask.shape
        xs = np.arange(w)
        need = w * 0.06                          # минимум ширины трассы в строке
        # ищем самую верхнюю (дальнюю) строку, где трасса ещё уверенно видна
        for r in range(bh):
            block = mask[r:min(r + 4, bh)]       # усредняем несколько строк для устойчивости
            tot = block.sum()
            if tot > need * block.shape[0]:
                centroid = float((block.sum(axis=0) * xs).sum() / block.sum(axis=0).sum())
                return float(np.clip((centroid - w / 2.0) / (w / 2.0), -1.0, 1.0)), \
                    float(mask.mean())
        return 0.0, float(mask.mean())

    def gray_strip(self, frame: np.ndarray) -> np.ndarray:
        """Серый центральный вертикальный срез — для оценки скорости."""
        h, w = frame.shape[:2]
        y0 = int(h * self.band_top)
        y1 = int(h * self.band_bottom)
        x0 = int(w * 0.35)
        x1 = int(w * 0.65)
        s = self.proc_step
        return frame[y0:y1:s, x0:x1:s, :].mean(axis=2).astype(np.float32)

    @staticmethod
    def speed_proxy(prev: np.ndarray, now: np.ndarray, max_shift: int = 6) -> float:
        """
        Оценка скорости через вертикальный оптический поток: при движении вперёд
        текстура трассы "сползает" вниз по экрану. Ищем сдвиг (в строках), при
        котором кадры лучше всего совпадают. Возвращает 0..1 (доля от max_shift).
        """
        if prev is None or now is None or prev.shape != now.shape:
            return 0.0
        best_shift, best_sad = 0, float("inf")
        for s in range(0, max_shift + 1):
            if s == 0:
                a, b = now, prev
            else:
                a, b = now[s:], prev[:-s]
            sad = float(np.abs(a - b).mean())
            if sad < best_sad:
                best_sad, best_shift = sad, s
        return best_shift / max_shift if max_shift else 0.0

    def debug_ascii(self, frame: np.ndarray, width: int = 60) -> str:
        """Текстовая визуализация того, что видит бот (линия G/Y/R или асфальт #)."""
        h, w = frame.shape[:2]
        y0 = int(h * self.band_top)
        y1 = int(h * self.band_bottom)
        band = frame[y0:y1, :, :].astype(np.int16)
        rstep = max(1, band.shape[0] // 12)
        cstep = max(1, w // width)
        if self.mode == "racing_line":
            green, yellow, red = self._line_masks(band)
            g = green[::rstep, ::cstep]; y = yellow[::rstep, ::cstep]; r = red[::rstep, ::cstep]
            rows = []
            for ri in range(g.shape[0]):
                line = "".join("R" if r[ri, ci] else "Y" if y[ri, ci] else
                               "G" if g[ri, ci] else "." for ci in range(g.shape[1]))
                rows.append(line)
            return "\n".join(rows)
        mask = self._segment(band) > 0.5
        small = mask[::rstep, ::cstep]
        return "\n".join("".join("#" if c else "." for c in row) for row in small)
