"""
Claude Vision — гоночный инженер/инструктор по телеметрии.

Снимает несколько кадров окна Roblox (HUD, трасса, поведение машины), шлёт их в
Claude (модель claude-opus-4-8) через официальный SDK `anthropic` и получает:
  • Анализ стиля пилотирования (имитация);
  • Конкретные корректировки (где раньше тормозить, агрессивнее выход и т.д.);
  • Config Update — JSON с настройками авто (давление шин, подвеска, передачи);
  • Сравнительную таблицу телеметрии (игрок vs оптимал, дельта);
  • autopilot_tuning — параметры нашего бота, которые сразу применяются.

Best-practice по навыку claude-api: vision через base64-картинки, adaptive
thinking, streaming для длинного ответа, prompt caching на стабильном
системном промпте (персона инженера).
"""

import base64
import json
import os
import re
import time

import mss
import mss.tools
import numpy as np

from paths import data_dir
from window import window_region

MODEL_DEFAULT = "claude-opus-4-8"

SYSTEM_PROMPT = """\
Ты — эксперт-инженер по телеметрии и профессиональный гоночный инструктор в \
симуляторе Formula Apex Racing (Roblox). Тебе присылают кадры заезда: на них \
виден HUD (спидометр, передача, обороты, температуры шин/тормозов, ERS/DRS), \
трасса, точки торможения, апексы, траектория и поведение машины (крен, снос, занос).

Твои задачи:
1. АНАЛИЗ СТИЛЯ (имитация): как пилот проходит повороты — точки торможения, вход, \
апекс, выход, работа газом и батареей. Сформируй краткий «цифровой профиль» стиля.
2. ОПТИМИЗАЦИЯ: сравни с идеальной гоночной траекторией. Где теряется время. \
Дай конкретные действия («в 4-м повороте тормози на ~10 м позже», «агрессивнее \
выход», «позже переключай передачу»).
3. КОНФИГУРАЦИЯ авто под этот стиль (давление шин, жёсткость подвески, \
передаточные числа, развал, антикрыло).
4. ТЕЛЕМЕТРИЯ: сравнительная таблица (скорость в апексе, время/точка торможения, \
обороты на выходе): игрок vs оптимал vs дельта.

ФОРМАТ ОТВЕТА (строго, на русском):

## Анализ
<кратко что увидел: профиль стиля>

## Корректировки
- <действие 1>
- <действие 2>
...

## Сравнение (телеметрия)
| Параметр | Игрок | Оптимал | Дельта |
|---|---|---|---|
| ... | ... | ... | ... |

## Config Update
```json
{
  "name": "<короткое имя сетапа>",
  "car_setup": {
    "tire_pressure_front": <число, psi>,
    "tire_pressure_rear": <число, psi>,
    "suspension_stiffness_front": <0..100>,
    "suspension_stiffness_rear": <0..100>,
    "gear_ratio": "<short|medium|long>",
    "front_wing": <0..50>,
    "rear_wing": <0..50>,
    "brake_bias": <50..70, % вперёд>
  },
  "autopilot_tuning": {
    "brake_error": <0.35..0.8, порог торможения>,
    "lift_error": <0.15..0.5, порог сброса газа>,
    "lookahead": <0.0..0.85, насколько смотреть вперёд>,
    "corner_brake": <0.4..2.0, упреждение торможения по скорости>,
    "line_weight": <0.0..1.0, агрессивность гоночной линии/апекса>
  }
}
```

Если на кадрах чего-то не видно — честно скажи и дай оценку по тому, что видно. \
autopilot_tuning подбирай так, чтобы бот ехал быстрее и чище по гоночной линии."""


class RaceCoach:
    def __init__(self, cfg, autopilot=None):
        self.cfg = cfg
        self.ap = autopilot
        c = cfg.get("claude", {})
        self.model = c.get("model", MODEL_DEFAULT)
        self.frames = int(c.get("frames", 5))
        self.frame_interval = float(c.get("frame_interval", 0.6))
        self.image_width = int(c.get("image_width", 900))
        self.apply_tuning = bool(c.get("apply_tuning", True))
        self.window_title = cfg.get("capture", {}).get("window_title", "Roblox")
        self._api_key = c.get("api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        self.busy = False
        self.reports_dir = os.path.join(data_dir(), "reports")
        self.setups_dir = os.path.join(data_dir(), "setups")
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.setups_dir, exist_ok=True)

    # ---- захват кадров с экрана ----------------------------------------
    def _region(self):
        return window_region(self.window_title) or {"top": 0, "left": 0,
                                                     "width": 1920, "height": 1080}

    def capture_frames(self):
        """Снимает серию кадров окна Roblox и кодирует в PNG (уменьшенные)."""
        region = self._region()
        sct = mss.mss()
        step = max(1, region["width"] // self.image_width)
        pngs = []
        for i in range(self.frames):
            arr = np.asarray(sct.grab(region))           # BGRA
            small = arr[::step, ::step]
            rgb = np.ascontiguousarray(small[:, :, [2, 1, 0]])   # BGR->RGB
            size = (rgb.shape[1], rgb.shape[0])
            png = mss.tools.to_png(rgb.tobytes(), size)
            pngs.append(png)
            if i < self.frames - 1:
                time.sleep(self.frame_interval)
        return pngs

    # ---- запрос к Claude ------------------------------------------------
    def analyze(self, log=print):
        if self.busy:
            log("[Claude] Уже анализирую, подожди...")
            return None
        if not self._api_key:
            log("[Claude] Нет ключа. Установи переменную ANTHROPIC_API_KEY или "
                "впиши claude.api_key в config.json.")
            return None
        try:
            import anthropic
        except ImportError:
            log("[Claude] Не установлен пакет. Выполни: pip install anthropic")
            return None

        self.busy = True
        try:
            log(f"[Claude] Снимаю {self.frames} кадров заезда...")
            pngs = self.capture_frames()

            content = []
            for png in pngs:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png",
                               "data": base64.standard_b64encode(png).decode()},
                })
            content.append({"type": "text", "text": self._user_ask()})

            log("[Claude] Отправляю кадры в Claude (opus-4-8, vision)...")
            client = anthropic.Anthropic(api_key=self._api_key)
            with client.messages.stream(
                model=self.model,
                max_tokens=8000,
                thinking={"type": "adaptive"},
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
            ) as stream:
                msg = stream.get_final_message()

            report = "".join(b.text for b in msg.content if b.type == "text")
            self._save_report(report)
            applied = self._apply_json(report, log)
            log("\n" + "=" * 60 + "\n" + report + "\n" + "=" * 60)
            if applied:
                log(f"[Claude] Применил тюнинг бота: {applied}")
            u = msg.usage
            log(f"[Claude] Готово. Токены: вход {u.input_tokens} "
                f"(кэш {getattr(u, 'cache_read_input_tokens', 0)}), выход {u.output_tokens}.")
            return report
        except Exception as e:
            log(f"[Claude] Ошибка: {e}")
            return None
        finally:
            self.busy = False

    def _user_ask(self):
        ctx = ""
        if self.ap is not None and getattr(self.ap, "lap", None) is not None:
            lp = self.ap.lap
            ctx = (f"\nКонтекст от бота: лучший круг={lp._fmt(lp.best_lap)}, "
                   f"изучено трассы={lp.known_fraction() * 100:.0f}%, "
                   f"опасных мест(срезы)={len(lp.cuts)}.")
        return ("Проанализируй мой заезд по этим кадрам и дай отчёт строго в "
                "заданном формате (Анализ, Корректировки, Сравнение, Config Update "
                "с JSON)." + ctx)

    # ---- сохранение и применение ---------------------------------------
    def _save_report(self, report):
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.reports_dir, f"report_{ts}.md")
        with open(path, "w", encoding="utf-8") as f:
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
                val = max(lo, min(hi, float(v)))
                if k in brain.int_params:
                    val = int(round(val))
                brain.best_params[k] = val
                self.ap.p[k] = val
                applied[k] = val
        if applied:
            brain.save()
        return applied or None
