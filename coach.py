"""
Vision-разбор заезда: гоночный инженер/инструктор по телеметрии.

Снимает кадры окна Roblox (HUD, трасса, поведение машины) и отправляет в
мультимодальную модель — Gemini (по умолчанию, бесплатно) или Claude — получая:
  • Анализ стиля пилотирования (имитация);
  • Корректировки (где раньше тормозить, агрессивнее выход и т.д.);
  • Config Update — JSON с настройками авто;
  • Сравнительную таблицу телеметрии (игрок vs оптимал vs дельта);
  • autopilot_tuning — параметры нашего бота, которые сразу применяются.

Провайдер выбирается в config.json -> coach.provider ("gemini" | "anthropic").
Gemini работает через REST (urllib, без зависимостей). Claude — через SDK anthropic.
"""

import base64
import json
import os
import re
import time
import urllib.request

import mss
import mss.tools
import numpy as np

from paths import data_dir
from window import window_region

SYSTEM_PROMPT = """\
Ты — эксперт-инженер по телеметрии и профессиональный гоночный инструктор в \
симуляторе Formula Apex Racing (Roblox). Тебе присылают кадры заезда: на них \
виден HUD (спидометр, передача, обороты, температуры шин/тормозов, ERS/DRS), \
трасса, точки торможения, апексы, траектория и поведение машины (крен, снос, занос).

Твои задачи:
1. АНАЛИЗ СТИЛЯ (имитация): как пилот проходит повороты — точки торможения, вход, \
апекс, выход, работа газом и батареей. Сформируй краткий «цифровой профиль» стиля.
2. ОПТИМИЗАЦИЯ: сравни с идеальной гоночной траекторией. Где теряется время. \
Дай конкретные действия («в 4-м повороте тормози на ~10 м позже», «агрессивнее выход»).
3. КОНФИГУРАЦИЯ авто под этот стиль (давление шин, жёсткость подвески, передачи, антикрыло).
4. ТЕЛЕМЕТРИЯ: сравнительная таблица (скорость в апексе, точка торможения, обороты на выходе): \
игрок vs оптимал vs дельта.

ФОРМАТ ОТВЕТА (строго, на русском):

## Анализ
<профиль стиля>

## Корректировки
- <действие 1>
- <действие 2>

## Сравнение (телеметрия)
| Параметр | Игрок | Оптимал | Дельта |
|---|---|---|---|
| ... | ... | ... | ... |

## Config Update
```json
{
  "name": "<имя сетапа>",
  "car_setup": {
    "tire_pressure_front": <psi>, "tire_pressure_rear": <psi>,
    "suspension_stiffness_front": <0..100>, "suspension_stiffness_rear": <0..100>,
    "gear_ratio": "<short|medium|long>", "front_wing": <0..50>, "rear_wing": <0..50>,
    "brake_bias": <50..70>
  },
  "autopilot_tuning": {
    "brake_error": <0.35..0.8>, "lift_error": <0.15..0.5>, "lookahead": <0.0..0.85>,
    "corner_brake": <0.4..2.0>, "line_weight": <0.0..1.0>
  }
}
```
Если чего-то не видно — честно скажи. autopilot_tuning подбирай так, чтобы бот ехал \
быстрее и чище по гоночной линии."""


class RaceCoach:
    def __init__(self, cfg, autopilot=None):
        self.cfg = cfg
        self.ap = autopilot
        c = cfg.get("coach") or cfg.get("claude") or {}
        self.provider = c.get("provider", "gemini").lower()
        self.frames = int(c.get("frames", 5))
        self.frame_interval = float(c.get("frame_interval", 0.6))
        self.image_width = int(c.get("image_width", 900))
        self.apply_tuning = bool(c.get("apply_tuning", True))
        self.gemini_model = c.get("gemini_model", "gemini-2.0-flash")
        self.anthropic_model = c.get("anthropic_model", "claude-opus-4-8")
        self.gemini_key = c.get("gemini_api_key", "") or os.environ.get("GEMINI_API_KEY", "") \
            or os.environ.get("GOOGLE_API_KEY", "")
        self.anthropic_key = c.get("anthropic_api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        self.window_title = cfg.get("capture", {}).get("window_title", "Roblox")
        self.busy = False
        self.reports_dir = os.path.join(data_dir(), "reports")
        self.setups_dir = os.path.join(data_dir(), "setups")
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.setups_dir, exist_ok=True)

    # ---- захват кадров --------------------------------------------------
    def capture_frames(self):
        region = window_region(self.window_title) or {"top": 0, "left": 0,
                                                       "width": 1920, "height": 1080}
        sct = mss.mss()
        step = max(1, region["width"] // self.image_width)
        out = []
        for i in range(self.frames):
            arr = np.asarray(sct.grab(region))
            small = arr[::step, ::step]
            rgb = np.ascontiguousarray(small[:, :, [2, 1, 0]])
            png = mss.tools.to_png(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]))
            out.append(base64.standard_b64encode(png).decode())
            if i < self.frames - 1:
                time.sleep(self.frame_interval)
        return out

    # ---- основной разбор ------------------------------------------------
    def analyze(self, log=print):
        if self.busy:
            log("[Разбор] Уже анализирую...")
            return None
        self.busy = True
        try:
            log(f"[Разбор] Провайдер: {self.provider}. Снимаю {self.frames} кадров...")
            imgs = self.capture_frames()
            ask = self._user_ask()
            if self.provider == "anthropic":
                report = self._call_anthropic(imgs, ask, log)
            else:
                report = self._call_gemini(imgs, ask, log)
            if not report:
                return None
            self._save_report(report)
            applied = self._apply_json(report, log)
            log("\n" + "=" * 60 + "\n" + report + "\n" + "=" * 60)
            if applied:
                log(f"[Разбор] Применил тюнинг бота: {applied}")
            return report
        except Exception as e:
            log(f"[Разбор] Ошибка: {e}")
            return None
        finally:
            self.busy = False

    # ---- Gemini (REST, без зависимостей) -------------------------------
    def _call_gemini(self, imgs, ask, log):
        if not self.gemini_key:
            log("[Разбор] Нет ключа Gemini. Установи GEMINI_API_KEY или впиши "
                "coach.gemini_api_key в config.json. Ключ бесплатно: aistudio.google.com/apikey")
            return None
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.gemini_model}:generateContent?key={self.gemini_key}")
        parts = [{"inline_data": {"mime_type": "image/png", "data": b}} for b in imgs]
        parts.append({"text": ask})
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": 8000, "temperature": 0.7},
        }
        log(f"[Разбор] Отправляю кадры в Gemini ({self.gemini_model})...")
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            log(f"[Разбор] Gemini HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}")
            return None
        cands = data.get("candidates") or []
        if not cands:
            log(f"[Разбор] Gemini не вернул ответ (возможно фильтр): {str(data)[:200]}")
            return None
        return "".join(p.get("text", "") for p in cands[0]["content"]["parts"])

    # ---- Claude (SDK anthropic) ----------------------------------------
    def _call_anthropic(self, imgs, ask, log):
        if not self.anthropic_key:
            log("[Разбор] Нет ключа Anthropic. Установи ANTHROPIC_API_KEY или "
                "coach.anthropic_api_key, либо переключи coach.provider на 'gemini'.")
            return None
        try:
            import anthropic
        except ImportError:
            log("[Разбор] Нет пакета anthropic. Выполни: pip install anthropic")
            return None
        content = [{"type": "image", "source": {"type": "base64",
                    "media_type": "image/png", "data": b}} for b in imgs]
        content.append({"type": "text", "text": ask})
        log(f"[Разбор] Отправляю кадры в Claude ({self.anthropic_model})...")
        client = anthropic.Anthropic(api_key=self.anthropic_key)
        with client.messages.stream(
            model=self.anthropic_model, max_tokens=8000,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
        ) as stream:
            msg = stream.get_final_message()
        return "".join(b.text for b in msg.content if b.type == "text")

    # ---- общее ----------------------------------------------------------
    def _user_ask(self):
        ctx = ""
        if self.ap is not None and getattr(self.ap, "lap", None) is not None:
            lp = self.ap.lap
            ctx = (f"\nКонтекст от бота: лучший круг={lp._fmt(lp.best_lap)}, "
                   f"изучено трассы={lp.known_fraction() * 100:.0f}%, "
                   f"опасных мест={len(lp.cuts)}.")
        return ("Проанализируй мой заезд по кадрам и дай отчёт строго в заданном "
                "формате (Анализ, Корректировки, Сравнение, Config Update с JSON)." + ctx)

    def _save_report(self, report):
        ts = time.strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(self.reports_dir, f"report_{ts}.md"), "w",
                  encoding="utf-8") as f:
            f.write(report)

    def _apply_json(self, report, log):
        m = re.search(r"```json\s*(\{.*?\})\s*```", report, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(1))
        except Exception:
            return None
        ts = time.strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(self.setups_dir, f"setup_{ts}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tuning = data.get("autopilot_tuning") or {}
        if not (self.apply_tuning and tuning and self.ap is not None):
            return None
        brain = self.ap.brain
        applied = {}
        for k, v in tuning.items():
            if k in brain.bounds:
                lo, hi = brain.bounds[k]
                try:
                    val = max(lo, min(hi, float(v)))
                except (TypeError, ValueError):
                    continue
                if k in brain.int_params:
                    val = int(round(val))
                brain.best_params[k] = val
                self.ap.p[k] = val
                applied[k] = val
        if applied:
            brain.save()
        return applied or None
