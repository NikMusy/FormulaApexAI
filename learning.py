"""
Обучение автопилота через эволюцию (1+1 Evolution Strategy).

Идея простая и устойчивая:
  * есть лучший набор параметров езды (best_params) и его оценка (best_score);
  * на каждый "эпизод" обучения создаём кандидата = лучший + случайный шум;
  * катаемся кандидатом несколько секунд, считаем средний reward
    (быстро + держится на трассе + по центру = хорошо; вылетел = штраф);
  * если кандидат лучше — он становится новым лучшим. Иначе откатываемся.

Лучший "мозг" сохраняется в brain.json и улучшается от заезда к заезду.
"""

import json
import os
import random
import time

import numpy as np

BRAIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain.json")
DEMO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos.json")


class Brain:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bounds = cfg["bounds"]
        self.int_params = {"ers_straight_frames", "drs_straight_frames", "max_step"}
        self._load(cfg["params"])

    # ---- сохранение / загрузка -----------------------------------------
    def _load(self, defaults):
        if os.path.exists(BRAIN_PATH):
            with open(BRAIN_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            # дополняем недостающие параметры дефолтами (совместимость со старым мозгом)
            self.best_params = {**defaults, **d.get("best_params", {})}
            self.best_score = d.get("best_score", None)
            self.generation = d.get("generation", 0)
            self.episodes = d.get("episodes", 0)
            self.history = d.get("history", [])
            print(f"[мозг] Загружен: поколение {self.generation}, "
                  f"лучший счёт {self._fmt(self.best_score)}, эпизодов {self.episodes}")
        else:
            self.best_params = dict(defaults)
            self.best_score = None
            self.generation = 0
            self.episodes = 0
            self.history = []
            print("[мозг] Новый — начинаю учиться с нуля.")

    def save(self):
        d = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "generation": self.generation,
            "episodes": self.episodes,
            "history": self.history[-100:],
        }
        with open(BRAIN_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    # ---- эволюция -------------------------------------------------------
    def make_candidate(self):
        """Первый эпизод — оцениваем самого лучшего (базлайн); далее — мутация."""
        if self.best_score is None:
            return dict(self.best_params), False
        scale = self.cfg["learning"]["mutation_scale"]
        cand = {}
        for k, v in self.best_params.items():
            lo, hi = self.bounds[k]
            sigma = (hi - lo) * scale
            nv = v + random.gauss(0, sigma)
            nv = max(lo, min(hi, nv))
            if k in self.int_params:
                nv = int(round(nv))
            cand[k] = nv
        return cand, True

    def report(self, candidate, score, mutated):
        self.episodes += 1
        improved = False
        if self.best_score is None or score > self.best_score:
            self.best_params = dict(candidate)
            self.best_score = score
            improved = True
            if mutated:
                self.generation += 1
        self.history.append({
            "t": time.strftime("%H:%M:%S"),
            "score": round(score, 3),
            "best": round(self.best_score, 3),
            "improved": improved,
            "gen": self.generation,
        })
        self.save()
        return improved

    # ---- обучение ПОКАЗОМ (имитация манеры игрока) ---------------------
    def learn_from_demo(self, samples, persist=True, reset_score=True, quiet=False):
        """
        samples: список dict с полями error, conf, speed, offset (смещение курсора
        от центра в px), throttle, brake, ers — записанными, пока ехал игрок.
        Извлекаем манеру: чувствительность руля (gain), пороги газа/тормоза, ERS.
          persist     — дописать ли образцы в demos.json (накопление между сессиями);
          reset_score — обнулить счёт, чтобы эволюция (F6) переоценила вокруг манеры;
          quiet       — без подробного вывода (для фонового обучения).
        """
        mincov = self.cfg["vision"]["min_coverage"]
        on = [s for s in samples if s["conf"] >= mincov]
        if len(on) < 30:
            if not quiet:
                print(f"[показ] Мало данных на трассе ({len(on)} кадров) — не учу.")
            return False

        err = np.array([s["error"] for s in on], dtype=np.float64)
        off = np.array([s["offset"] for s in on], dtype=np.float64)
        ae = np.abs(err)
        thr = np.array([s["throttle"] for s in on], dtype=bool)
        brk = np.array([s["brake"] for s in on], dtype=bool)
        ers = np.array([s["ers"] for s in on], dtype=bool)

        p = dict(self.best_params)
        learned = []

        # Чувствительность руля: наклон offset(px) по error (линия через 0).
        denom = float((err ** 2).sum())
        if denom > 1e-3 and float(np.abs(off).mean()) > 5.0:
            gain = abs(float((err * off).sum() / denom))
            p["gain"] = self._clip("gain", gain)
            learned.append(f"gain={p['gain']:.0f}")

        # Порог торможения: типичная ошибка, при которой игрок тормозил.
        if brk.sum() >= 8:
            p["brake_error"] = self._clip("brake_error", float(np.percentile(ae[brk], 35)))
            learned.append(f"brake_error={p['brake_error']:.2f}")

        # Порог сброса газа: ошибка, когда игрок не газовал и не тормозил.
        coast = (~thr) & (~brk)
        if coast.sum() >= 8:
            le = float(np.percentile(ae[coast], 50))
            p["lift_error"] = self._clip("lift_error", min(le, p["brake_error"] - 0.03))
            learned.append(f"lift_error={p['lift_error']:.2f}")

        # ERS: насколько ровно ехал игрок, когда жал батарею -> порог "прямой".
        if ers.sum() >= 5:
            se = float(np.percentile(ae[ers], 70))
            p["straight_error"] = self._clip("straight_error", se)
            learned.append(f"straight_error={p['straight_error']:.2f}")

        if not learned:
            if not quiet:
                print("[показ] Не удалось извлечь манеру (мало движения/нажатий).")
            return False

        self.best_params = p
        if reset_score:
            self.best_score = None    # пусть эволюция переоценит и оттачивает
        if persist:
            self.append_demos(samples)
        self.save()
        if not quiet:
            print(f"[показ] Запомнил манеру ({len(on)} кадров): {', '.join(learned)}. "
                  "Жми F6 — оттачивать, или F8 — ехать так.")
        else:
            print(f"\n[фон] дообучился на {len(on)} кадрах: {', '.join(learned)}")
        return True

    def _clip(self, key, value):
        lo, hi = self.bounds[key]
        value = max(lo, min(hi, value))
        return int(round(value)) if key in self.int_params else value

    def load_all_demos(self):
        if os.path.exists(DEMO_PATH):
            try:
                with open(DEMO_PATH, "r", encoding="utf-8") as f:
                    return json.load(f).get("samples", [])
            except Exception:
                return []
        return []

    def append_demos(self, samples):
        data = {"samples": (self.load_all_demos() + list(samples))[-120000:]}
        with open(DEMO_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @staticmethod
    def _fmt(x):
        return "—" if x is None else f"{x:.3f}"
