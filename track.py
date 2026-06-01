"""
Карта трассы и круги: запоминает где старт/финиш, длину круга, время круга
(лучшее), и места corner cut / вылетов. Эти данные копятся в track_map.json
и используются, чтобы бот САМ притормаживал в опасных местах и улучшал время.

Дистанция считается интегралом скорости (оптический поток) по времени — точной
геометрии нет, но для разметки "где по кругу мы сейчас" этого достаточно.
"""

import json
import os
import time

from paths import data_dir

TRACK_PATH = os.path.join(data_dir(), "track_map.json")


class LapTracker:
    def __init__(self, cfg):
        b = cfg["behavior"]
        self.enabled = b.get("track_map", True)
        self.cut_speed = b.get("cut_speed", 0.15)
        self.danger_window = b.get("danger_lookahead", 0.05)
        self._load()
        self.reset_run()

    # ---- сохранение / загрузка -----------------------------------------
    def _load(self):
        self.N = 240                 # на сколько отрезков делим круг (память трассы)
        self.lap_length = None
        self.best_lap = None
        self.cuts = []               # доли круга 0..1, где были срезы/вылеты
        self.cnt = [0] * self.N      # сколько раз проезжали отрезок
        self.scurv = [0.0] * self.N  # сумма кривизны на отрезке
        self.snear = [0.0] * self.N  # сумма "куда идёт трасса" на отрезке
        self.hoff_cnt = [0] * self.N    # сколько раз ИГРОК ехал этот отрезок
        self.hoff_sum = [0.0] * self.N  # сумма смещения руля ИГРОКА (px) -> его траектория
        self.best_off = [None] * self.N # ИДЕАЛЬНАЯ линия: смещение руля на самом быстром круге
        self.best_off_lap = None        # время того самого быстрого круга
        if os.path.exists(TRACK_PATH):
            try:
                with open(TRACK_PATH, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.lap_length = d.get("lap_length")
                self.best_lap = d.get("best_lap")
                self.cuts = d.get("cuts", [])
                if d.get("N") == self.N and len(d.get("cnt", [])) == self.N:
                    self.cnt = d["cnt"]; self.scurv = d["scurv"]; self.snear = d["snear"]
                    if len(d.get("hoff_cnt", [])) == self.N:
                        self.hoff_cnt = d["hoff_cnt"]; self.hoff_sum = d["hoff_sum"]
                    if len(d.get("best_off", [])) == self.N:
                        self.best_off = d["best_off"]; self.best_off_lap = d.get("best_off_lap")
                known = sum(1 for c in self.cnt if c > 0)
                print(f"[трасса] Карта загружена: лучший круг={self._fmt(self.best_lap)}, "
                      f"изучено отрезков={known}/{self.N}, опасных мест={len(self.cuts)}")
            except Exception:
                pass

    def save(self):
        with open(TRACK_PATH, "w", encoding="utf-8") as f:
            json.dump({"lap_length": self.lap_length, "best_lap": self.best_lap,
                       "cuts": self.cuts[-200:], "N": self.N,
                       "cnt": self.cnt, "scurv": self.scurv, "snear": self.snear,
                       "hoff_cnt": self.hoff_cnt, "hoff_sum": self.hoff_sum,
                       "best_off": self.best_off, "best_off_lap": self.best_off_lap},
                      f, ensure_ascii=False)

    def reset_run(self):
        now = time.perf_counter()
        self.distance = 0.0
        self.lap_start = now
        self._last_t = now
        self._off = False
        self.anchored = False    # отмечена ли стартовая линия (F4)
        self.laps = 0
        self.last_lap = None
        self.cut_count = 0
        self._pending_cuts = []  # сырые дистанции срезов текущего круга
        self._loff_sum = [0.0] * self.N   # траектория руля ТЕКУЩЕГО круга
        self._loff_cnt = [0] * self.N

    # ---- разметка линии старт/финиш (F4) -------------------------------
    def mark_line(self):
        now = time.perf_counter()
        if not self.anchored:
            self.anchored = True
            self.distance = 0.0
            self.lap_start = now
            return ("start", None)
        return ("lap", self._complete_lap(now))

    def _complete_lap(self, now):
        lap_time = now - self.lap_start
        if lap_time < 3.0:                # защита от случайных двойных нажатий
            return None
        if self.lap_length is None:
            self.lap_length = self.distance
        else:
            self.lap_length = 0.7 * self.lap_length + 0.3 * self.distance
        self.last_lap = lap_time
        self.laps += 1
        improved = self.best_lap is None or lap_time < self.best_lap
        if improved:
            self.best_lap = lap_time
            # ЗАПОМИНАЕМ ИДЕАЛЬНУЮ ЛИНИЮ: траекторию руля этого (рекордного) круга
            for i in range(self.N):
                if self._loff_cnt[i] > 0:
                    self.best_off[i] = self._loff_sum[i] / self._loff_cnt[i]
            self.best_off_lap = lap_time
        # нормализуем срезы этого круга в доли 0..1 и запоминаем
        if self.lap_length and self.lap_length > 0:
            for d in self._pending_cuts:
                self.cuts.append(round(min(0.999, d / self.lap_length), 3))
        self._pending_cuts = []
        self._loff_sum = [0.0] * self.N
        self._loff_cnt = [0] * self.N
        self.save()
        self.distance = 0.0
        self.lap_start = now
        return {"time": lap_time, "best": self.best_lap, "improved": improved}

    # ---- обновление каждый кадр ----------------------------------------
    def update(self, speed, on_track, curv=0.0, near=0.0, offset=None):
        """Возвращает dict событий: {'cut': frac} и/или {'lap': {...}}."""
        if not self.enabled:
            return {}
        now = time.perf_counter()
        dt = now - self._last_t
        self._last_t = now
        self.distance += speed * dt
        events = {}

        # ЗАПОМИНАЕМ форму трассы на этом отрезке круга (карта по памяти)
        f = self.frac()
        if f is not None and on_track:
            i = min(self.N - 1, int(f * self.N))
            self.cnt[i] += 1
            self.scurv[i] += curv
            self.snear[i] += near
            if offset is not None:        # траектория руля текущего круга (для идеальной линии)
                self._loff_sum[i] += offset
                self._loff_cnt[i] += 1

        # corner cut / вылет: съехали с асфальта на скорости
        if not on_track and speed > self.cut_speed:
            if not self._off:
                self._off = True
                self._pending_cuts.append(self.distance)   # нормализуем в конце круга
                self.cut_count += 1
                events["cut"] = f                           # может быть None на 1-м круге
        elif on_track:
            self._off = False

        # авто-завершение круга по дистанции (после калибровки одним кругом)
        if self.anchored and self.lap_length and self.distance >= self.lap_length:
            lap = self._complete_lap(now)
            if lap:
                events["lap"] = lap
        return events

    # ---- запросы для управления ----------------------------------------
    def frac(self):
        """Текущая доля круга 0..1 (None, если не откалибровано)."""
        if not self.anchored or not self.lap_length:
            return None
        return min(0.999, self.distance / self.lap_length)

    def map_ahead(self, ahead=0.03, min_count=3):
        """
        Память трассы чуть ВПЕРЕДИ текущей позиции: что там за поворот.
        Возвращает {'curv', 'near', 'count'} или None, если участок не изучен.
        """
        f = self.frac()
        if f is None:
            return None
        i = int(((f + ahead) % 1.0) * self.N) % self.N
        c = self.cnt[i]
        if c < min_count:
            return None
        return {"curv": self.scurv[i] / c, "near": self.snear[i] / c, "count": c}

    def note_human(self, offset):
        """Запоминаем смещение руля ИГРОКА на текущем отрезке круга (его траекторию)."""
        f = self.frac()
        if f is None:
            return
        i = min(self.N - 1, int(f * self.N))
        self.hoff_cnt[i] += 1
        self.hoff_sum[i] += offset

    def human_offset_ahead(self, ahead=0.02, min_count=3):
        """Смещение руля, как ехал ИГРОК, чуть впереди (px). None, если не изучено."""
        f = self.frac()
        if f is None:
            return None
        i = int(((f + ahead) % 1.0) * self.N) % self.N
        c = self.hoff_cnt[i]
        if c < min_count:
            return None
        return self.hoff_sum[i] / c

    def ideal_offset_ahead(self, ahead=0.02):
        """Смещение руля на ИДЕАЛЬНОЙ (самой быстрой) линии чуть впереди. None если нет."""
        f = self.frac()
        if f is None:
            return None
        i = int(((f + ahead) % 1.0) * self.N) % self.N
        return self.best_off[i]

    def curv_window(self, frac_window, min_count=3):
        """Максимальная кривизна на участке ВПЕРЕДИ (смотрим дальше на скорости)."""
        f = self.frac()
        if f is None:
            return 0.0
        n = max(1, int(frac_window * self.N))
        start = int(f * self.N)
        mc = 0.0
        for k in range(1, n + 1):
            i = (start + k) % self.N
            if self.cnt[i] >= min_count:
                mc = max(mc, abs(self.scurv[i] / self.cnt[i]))
        return mc

    def ideal_known(self):
        return sum(1 for v in self.best_off if v is not None) / self.N

    def known_fraction(self):
        return sum(1 for c in self.cnt if c > 0) / self.N

    def danger_ahead(self):
        """True, если впереди (в пределах окна) есть запомненный corner cut."""
        f = self.frac()
        if f is None or not self.cuts:
            return False
        for c in self.cuts:
            d = (c - f) % 1.0
            if 0.0 <= d <= self.danger_window:
                return True
        return False

    @staticmethod
    def _fmt(t):
        if t is None:
            return "—"
        return f"{int(t // 60)}:{t % 60:06.3f}"
