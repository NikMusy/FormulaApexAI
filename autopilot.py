"""
Formula Apex AI — самообучающийся автопилот для Roblox: Formula Apex Racing.

Управление машиной:
  газ      = ЛКМ (зажата)
  тормоз   = ПКМ (зажата)
  батарея  = G (ERS / overtake)
  руль     = курсор мыши (абсолютное позиционирование — не "уезжает")

Горячие клавиши:
  F5 — ЗАПИСЬ: ты едешь сам, бот запоминает твою манеру (показ)
  F6 — ОБУЧЕНИЕ: бот оттачивает манеру эволюцией
  F8 — ЕЗДА на лучшем мозге
  F7 — калибровка зрения
  F9 — выход

Антибаг-меры: ввод трогает только главный поток; при потере трассы — пауза
и отпускание всех кнопок; курсор позиционируется абсолютно, без накопления.
"""

import json
import os
import time

import keyboard

import controller as C
from controller import Actuator, move_mouse, set_cursor, get_cursor, spec_pressed, key_pressed
from learning import Brain
from vision import TrackVision
from track import LapTracker
from window import set_dpi_aware
from paths import data_dir, resource_path

VK_G = 0x47


def load_cfg():
    # config.json: сначала рядом с приложением (можно редактировать), иначе упакованный
    path = os.path.join(data_dir(), "config.json")
    if not os.path.exists(path):
        path = resource_path("config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Autopilot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vision = TrackVision(cfg)
        self.act = Actuator()
        self.brain = Brain(cfg)
        self.lap = LapTracker(cfg)

        b = cfg["behavior"]
        self.keys = cfg["keys"]
        self.steer_mode = cfg["steering"]["mode"]
        self.use_ers = b["use_ers"]
        self.use_drs = b["use_drs"] and bool(self.keys["drs"])
        self.auto_line = b["auto_line"]
        self.always_learn = b["always_learn"]
        self.bg_train_seconds = b["bg_train_seconds"]
        self.bg_min_new = b["bg_min_new"]
        self.arm_seconds = b["arm_seconds"]
        self.detect_crash = b["detect_crash"]
        self.crash_speed = b["crash_speed"]
        self.crash_seconds = b["crash_seconds"]
        self.recover_seconds = b["recover_seconds"]
        self.max_brake_time = b.get("max_brake_time", 0.7)
        self.brake_cooldown = b.get("brake_cooldown", 0.5)
        self.brake_floor = b.get("brake_floor", 0.5)
        self.lift_floor = b.get("lift_floor", 0.3)
        self.map_weight = b.get("map_weight", 0.4)
        self.map_lookahead = b.get("map_lookahead", 0.025)
        self.map_min_count = b.get("map_min_count", 3)
        self.human_weight = b.get("human_line_weight", 0.5)
        self.ideal_weight = b.get("ideal_line_weight", 0.6)
        self.curv_window_base = b.get("curv_window_base", 0.03)
        self.curv_window_speed = b.get("curv_window_speed", 0.06)
        self.mem_frames = b.get("mem_frames", 60000)
        self.reset_key = self.keys.get("reset", "")
        self.target_dt = 1.0 / cfg["loop"]["target_fps"]

        lr = cfg["learning"]
        self.episode_seconds = lr["episode_seconds"]
        self.w_speed = lr["speed_weight"]
        self.w_center = lr["center_weight"]
        self.offtrack_penalty = lr["offtrack_penalty"]
        self.crash_penalty = lr["crash_penalty"]
        self.max_speed_shift = lr["max_speed_shift"]

        self._update_center()

        self.mode = "idle"
        self._pending = None          # команда от хоткея (тред-безопасно)
        self.p = dict(self.brain.best_params)

        self.candidate = None
        self.mutated = False
        self._reset_drive_state()
        self.demo = []
        self.arm_at = 0.0
        self._mark_line = False
        self.lost_frames = 0
        self.stop_flag = False

        # фоновое (бесконечное) обучение по тому, как едет игрок
        self.bg_buffer = list(self.brain.load_all_demos())[-cfg["behavior"].get("mem_frames", 60000):]
        self.bg_new = []
        self.last_bg_train = time.perf_counter()
        self.track_mem = 0.0          # сглаженная "память" где трасса
        if self.always_learn and len(self.bg_buffer) >= 100:
            print(f"[фон] Подгружаю твой стиль из {len(self.bg_buffer)} прошлых кадров...")
            self.brain.learn_from_demo(self.bg_buffer, persist=False,
                                       reset_score=False, quiet=False)
            self.p = dict(self.brain.best_params)

    # ---- центр захвата = центр руля (зависит от области окна) -----------
    def _update_center(self):
        r = self.vision.region
        self.cx = r["left"] + r["width"] // 2
        self.cy = r["top"] + r["height"] // 2

    # ---- состояние руления ---------------------------------------------
    def _reset_drive_state(self):
        self.smooth_error = 0.0
        self.smooth_off = 0.0
        self.straight_frames = 0
        self.prev_strip = None
        self.drive_state = "accel"    # accel | coast | brake
        self.avg_speed = 0.0
        self.curv = 0.0
        self.map_curv = 0.0
        self.map_human_off = None
        self.ideal_off = None
        self.last_conf = 0.0
        self.stuck_since = None
        self.recover_until = 0.0
        self.crashes = 0
        self.brake_started = 0.0
        self.brake_block_until = 0.0

    # ---- команды от хоткеев (только ставим флаг) -----------------------
    def request(self, mode):
        self._pending = mode

    def request_lap_line(self):
        self._mark_line = True

    def _handle_lap_line(self):
        if not self._mark_line:
            return
        self._mark_line = False
        kind, info = self.lap.mark_line()
        if kind == "start":
            print("\n[круг] Линия старт/финиш отмечена. Поезжай круг — замерю время.")
        elif kind == "lap" and info:
            d = ""
            if not info["improved"] and self.lap.best_lap:
                d = f"  (+{info['time'] - self.lap.best_lap:.3f} к рекорду)"
            tag = " РЕКОРД! 🏁" if info["improved"] else ""
            print(f"\n[круг] Круг: {self.lap._fmt(info['time'])}{tag}{d}  "
                  f"лучший={self.lap._fmt(info['best'])}")

    def _apply_pending_core(self):
        if self._pending is None:
            return
        mode = self._pending
        self._pending = None
        if self.mode == mode:
            mode = "idle"
        # выходим из текущего режима чисто
        self.act.release_all()
        self._reset_drive_state()
        self.lost_frames = 0
        self.paused = False
        self.mode = mode

        if mode == "idle":
            print("\n>>> Остановлено. Кнопки отпущены.")
            return

        # заново находим окно Roblox (вдруг подвинули) и пересчитываем центр руля
        self.vision.refresh_region()
        self._update_center()
        self.lap.reset_run()
        if self.vision.using_window:
            print(f"\n[окно] Захват Roblox {self.vision.region['width']}x"
                  f"{self.vision.region['height']}")
        else:
            print("\n[окно] !!! Окно Roblox НЕ найдено — захват всего монитора. "
                  "Открой игру и переключи режим заново.")

        self.arm_at = time.perf_counter() + self.arm_seconds
        if mode == "drive":
            self.p = dict(self.brain.best_params)
            print(f"\n>>> ЕЗДА. Кликни в окно Roblox — старт через {self.arm_seconds:.0f}с. F8 — стоп.")
        elif mode == "learn":
            print(f"\n>>> ОБУЧЕНИЕ. Кликни в окно Roblox — старт через {self.arm_seconds:.0f}с. F6 — стоп.")
            self._start_episode()
        elif mode == "record":
            self.demo = []
            print(f"\n>>> ЗАПИСЬ. Веди машину САМ — я запоминаю. F5 — стоп и обучение.")

    # ---- обучение эволюцией --------------------------------------------
    def _start_episode(self):
        self.candidate, self.mutated = self.brain.make_candidate()
        self.p = dict(self.candidate)
        self.reward_sum = 0.0
        self.reward_n = 0
        self.episode_start = time.perf_counter()
        self._reset_drive_state()

    def _finish_episode(self):
        score = self.reward_sum / self.reward_n if self.reward_n else 0.0
        improved = self.brain.report(self.candidate, score, self.mutated)
        mark = "ЛУЧШЕ ✓" if improved else "откат"
        print(f"\n[эпизод] счёт={score:6.3f} ({mark}) поколение={self.brain.generation} "
              f"рекорд={self.brain._fmt(self.brain.best_score)}")
        self._start_episode()

    # ---- зрение + общая телеметрия кадра -------------------------------
    def _perceive(self):
        frame = self.vision.grab()
        sc = self.vision.scene(frame)               # видим всю дорогу: near/mid/far/line/curv
        near, far, line, conf = sc["near"], sc["far"], sc["line"], sc["conf"]
        self.curv = sc["curv"]
        if not self.auto_line:
            line = 0.0
        strip = self.vision.gray_strip(frame)
        speed = self.vision.speed_proxy(self.prev_strip, strip, self.max_speed_shift)
        self.prev_strip = strip
        self.avg_speed = 0.7 * self.avg_speed + 0.3 * speed
        # "память" трассы: помним где она, даже когда на миг теряем из вида
        if conf >= self.vision.min_coverage:
            self.track_mem = 0.8 * self.track_mem + 0.2 * near
        else:
            self.track_mem *= 0.9
        return near, far, line, conf, speed

    # ---- фоновое наблюдение: учимся, пока игрок едет сам ----------------
    def _observe(self, now):
        near, far, line, conf, speed = self._perceive()
        self.last_conf = conf
        lmb = spec_pressed(self.keys["throttle"])
        rmb = spec_pressed(self.keys["brake"])
        driving = lmb or rmb or speed > self.crash_speed
        on_track = conf >= self.vision.min_coverage
        x, _ = get_cursor()
        h_off = float(x - self.cx) if (driving and on_track) else None
        # учимся структуре круга, форме трассы, местам срезов И твоей траектории
        self._report_lap_events(self.lap.update(
            speed, on_track=on_track, curv=self.curv, near=near, offset=h_off))
        if driving and on_track:
            self.lap.note_human(x - self.cx)        # запоминаем ТВОЮ траекторию по позиции
            s = {"error": float(near), "conf": float(conf), "speed": float(speed),
                 "offset": float(x - self.cx), "throttle": lmb, "brake": rmb,
                 "ers": key_pressed(VK_G)}
            self.bg_buffer.append(s)
            self.bg_new.append(s)
            if len(self.bg_buffer) > self.mem_frames:
                self.bg_buffer = self.bg_buffer[-self.mem_frames:]
        # периодически дообучаемся на накопленном
        if now - self.last_bg_train > self.bg_train_seconds and len(self.bg_new) >= self.bg_min_new:
            self.brain.learn_from_demo(self.bg_buffer[-4000:], persist=False,
                                       reset_score=True, quiet=True)
            self.brain.append_demos(self.bg_new)
            self.bg_new = []
            self.last_bg_train = now
        return conf, speed, driving

    # ---- руление: обзор вперёд + гоночная линия + ПАМЯТЬ трассы ---------
    def _steer(self, near_err, far_err, line_err, map_err=None):
        la = self.p["lookahead"]
        base = (1 - la) * near_err + la * far_err      # учитываем даль -> входим в поворот заранее
        lw = self.p["line_weight"] if self.auto_line else 0.0
        error = (1 - lw) * base + lw * line_err        # тянемся к дальней точке -> апекс
        if map_err is not None:                        # знаем этот участок по памяти всего круга
            error = (1 - self.map_weight) * error + self.map_weight * map_err
        self.smooth_error = (self.p["smoothing"] * self.smooth_error
                             + (1 - self.p["smoothing"]) * error)
        e = 0.0 if abs(self.smooth_error) < self.p["deadzone"] else self.smooth_error
        if self.steer_mode == "relative":
            dx = max(-self.p["max_step"], min(self.p["max_step"], e * self.p["gain"]))
            move_mouse(int(round(dx)), 0)
        else:  # absolute
            target = max(-self.p["max_step"], min(self.p["max_step"], e * self.p["gain"]))
            ms = self.p["max_step"]
            # ИДЕАЛЬНАЯ линия рекордного круга в приоритете; иначе — твоя средняя траектория
            if self.ideal_off is not None:
                ip = max(-ms, min(ms, self.ideal_off))
                target = (1 - self.ideal_weight) * target + self.ideal_weight * ip
            elif self.map_human_off is not None:
                hp = max(-ms, min(ms, self.map_human_off))
                target = (1 - self.human_weight) * target + self.human_weight * hp
            self.smooth_off = 0.5 * self.smooth_off + 0.5 * target
            set_cursor(self.cx + int(round(self.smooth_off)), self.cy)

    # ---- газ/тормоз: газ по умолчанию, тормоз РЕДКО (ПКМ=задний ход!) ----
    def _throttle(self, abs_near, abs_far, lost, now):
        m = 0.04
        # пороги не ниже "пола" — иначе тормозил бы постоянно и сдавал назад
        be = max(self.p["brake_error"], self.brake_floor)
        le = min(max(self.p["lift_error"], self.lift_floor), be - 0.05)

        # "предсказанная" опасность: даль + КРИВИЗНА впереди, сильнее на скорости
        danger = max(abs_near,
                     abs_far * self.p["lookahead"] + abs_near * (1 - self.p["lookahead"]),
                     abs(self.curv) * self.p["curve_brake"],
                     abs(self.map_curv) * self.p["curve_brake"])   # знаем поворот по памяти
        danger *= (0.6 + self.p["corner_brake"] * self.avg_speed)

        # запрет тормоза: после долгого тормоза остываем (чтобы ПКМ не ушёл в реверс)
        can_brake = now >= self.brake_block_until

        if lost:
            self.drive_state = "accel"
        elif self.drive_state == "accel":
            if danger > be and can_brake:
                self.drive_state = "brake"; self.brake_started = now
            elif danger > le:
                self.drive_state = "coast"
        elif self.drive_state == "coast":
            if danger > be and can_brake:
                self.drive_state = "brake"; self.brake_started = now
            elif danger < le - m:
                self.drive_state = "accel"
        elif self.drive_state == "brake":
            # держим ПКМ не дольше max_brake_time -> иначе реверс
            if now - self.brake_started > self.max_brake_time:
                self.drive_state = "coast"
                self.brake_block_until = now + self.brake_cooldown
            elif danger < le:
                self.drive_state = "accel"
            elif danger < be - m:
                self.drive_state = "coast"

        self.act.set(self.keys["throttle"], self.drive_state == "accel")
        self.act.set(self.keys["brake"], self.drive_state == "brake")

    # ---- детектор аварии / застревания ---------------------------------
    def _crash_check(self, now):
        """True, если машина пытается ехать, но не движется (удар/застряла)."""
        if not self.detect_crash:
            return False
        trying = self.drive_state == "accel"
        if trying and self.avg_speed < self.crash_speed:
            if self.stuck_since is None:
                self.stuck_since = now
            elif now - self.stuck_since > self.crash_seconds:
                return True
        else:
            self.stuck_since = None
        return False

    def _recover(self, now):
        """Восстановление: сброс машины (если есть клавиша) или сдать назад."""
        self.crashes += 1
        print(f"\n[АВАРИЯ #{self.crashes}] машина не едет — восстанавливаюсь...", end="")
        self.act.release_all()
        if self.reset_key:
            self.act.set(self.keys["throttle"], False)
            self.act.hold(self.reset_key); time.sleep(0.08); self.act.release(self.reset_key)
        self.recover_until = now + self.recover_seconds
        self.stuck_since = None

    # ---- один шаг автопилота (drive/learn) -----------------------------
    def step_auto(self, learning):
        now = time.perf_counter()
        near, far, line, conf, speed = self._perceive()
        lost = conf < self.vision.min_coverage * 1.2
        abs_near, abs_far = abs(near), abs(far)

        # идёт восстановление после аварии: сдаём назад и подруливаем
        if now < self.recover_until:
            set_cursor(self.cx - int(self.p["max_step"] * 0.5 * (1 if near >= 0 else -1)), self.cy)
            self.act.set(self.keys["throttle"], False)
            self.act.set(self.keys["brake"], True)        # ПКМ = назад
            self.act.set(self.keys["ers"], False)
            if learning:
                self.reward_sum += -self.crash_penalty; self.reward_n += 1
            return {"error": near, "conf": conf, "speed": speed, "crash": True}

        self.last_conf = conf
        # ПАМЯТЬ всего круга: что за поворот впереди на этом участке
        mp = self.lap.map_ahead(self.map_lookahead, self.map_min_count)
        map_err = mp["near"] if mp else None
        # смотрим дальше вперёд на скорости -> тормозим заранее перед поворотом
        window = self.curv_window_base + self.curv_window_speed * self.avg_speed * 10
        self.map_curv = self.lap.curv_window(window, self.map_min_count)
        # ИДЕАЛЬНАЯ линия рекорда (приоритет), иначе средняя траектория игрока
        self.ideal_off = self.lap.ideal_offset_ahead(self.map_lookahead)
        self.map_human_off = self.lap.human_offset_ahead(self.map_lookahead, self.map_min_count)

        # при потере трассы рулим по ПАМЯТИ (где она была), а не слепо в центр
        self._steer(self.track_mem if lost else near,
                    self.track_mem if lost else far,
                    self.track_mem if lost else line,
                    map_err)
        self._throttle(abs_near, abs_far, lost, now)

        # карта трассы + запись траектории руля бота этого круга (для идеальной линии)
        ev = self.lap.update(speed, on_track=not lost, curv=self.curv, near=near,
                             offset=self.smooth_off)
        self._report_lap_events(ev)

        # САМ притормаживает заранее в местах, где раньше срезал/вылетал
        if self.lap.danger_ahead() and self.drive_state == "accel":
            self.drive_state = "coast"
            self.act.set(self.keys["throttle"], False)

        if self._crash_check(now):
            self._recover(now)
            return {"error": near, "conf": conf, "speed": speed, "crash": True}

        if not lost and abs_near < self.p["straight_error"] and abs_far < self.p["straight_error"]:
            self.straight_frames += 1
        else:
            self.straight_frames = 0
        if self.use_ers:
            self.act.set(self.keys["ers"], self.straight_frames >= self.p["ers_straight_frames"])
        if self.use_drs:
            self.act.set(self.keys["drs"], self.straight_frames >= self.p["drs_straight_frames"])

        if learning:
            if lost:
                reward = -self.offtrack_penalty
            else:
                reward = self.w_speed * speed + self.w_center * (1.0 - abs_near)
            self.reward_sum += reward
            self.reward_n += 1
        return {"error": near, "conf": conf, "speed": speed, "crash": False}

    def _report_lap_events(self, ev):
        if "cut" in ev:
            f = ev["cut"]
            where = f"на доле круга {f:.2f}" if f is not None else "(круг ещё не измерен)"
            print(f"\n[CORNER CUT] вылет с трассы {where} — запомнил.")
        if ev.get("lap"):
            lp = ev["lap"]
            tag = " РЕКОРД! 🏁" if lp["improved"] else ""
            print(f"\n[круг] {self.lap._fmt(lp['time'])}{tag}  "
                  f"лучший={self.lap._fmt(lp['best'])}")

    # ---- один шаг записи (едет игрок, мы только смотрим) ---------------
    def step_record(self):
        near, far, line, conf, speed = self._perceive()
        error = near
        x, _ = get_cursor()
        self.demo.append({
            "error": float(error), "conf": float(conf), "speed": float(speed),
            "offset": float(x - self.cx),
            "throttle": spec_pressed(self.keys["throttle"]),
            "brake": spec_pressed(self.keys["brake"]),
            "ers": key_pressed(VK_G),
        })
        return {"error": error, "conf": conf, "speed": speed}

    # ---- главный цикл ---------------------------------------------------
    def run(self):
        hk = self.cfg["hotkeys"]
        keyboard.add_hotkey(hk["drive"], lambda: self.request("drive"))
        keyboard.add_hotkey(hk["learn"], lambda: self.request("learn"))
        keyboard.add_hotkey(hk["record"], lambda: self.request("record"))
        keyboard.add_hotkey(hk["calibrate"], self.calibrate)
        keyboard.add_hotkey(hk["lap_line"], self.request_lap_line)

        print("=" * 62)
        print(" Formula Apex AI — самообучающийся автопилот")
        print("=" * 62)
        src = (f"окно Roblox {self.vision.region['width']}x{self.vision.region['height']}"
               if self.vision.using_window else "ВЕСЬ ЭКРАН (окно Roblox не найдено!)")
        print(f" Захват: {src}   руль: {self.steer_mode}")
        print(f" Газ=ЛКМ  Тормоз=ПКМ  Батарея(ERS)={self.keys['ers']}"
              + (f"  DRS={self.keys['drs']}" if self.use_drs else "  DRS=выкл"))
        print(f" {hk['record'].upper()}=ЗАПИСЬ  {hk['learn'].upper()}=обучение  "
              f"{hk['drive'].upper()}=езда  {hk['lap_line'].upper()}=старт/финиш круга  "
              f"{hk['calibrate'].upper()}=калибровка  {hk['quit'].upper()}=выход")
        print("=" * 62)
        if self.always_learn:
            print(" ФОН: просто играй сам — я постоянно смотрю и учусь твоей езде.")
            print(" Когда захочешь — жми F8, и я поеду сам (и буду оттачивать F6).")
        else:
            print(" План: F5 -> проедь круг сам -> F5 -> F6 (оттачивает) -> F8 (едет).")

        last_log = 0.0
        try:
            while not self.stop_flag:
                if keyboard.is_pressed(hk["quit"]):
                    break
                t0 = time.perf_counter()
                self._apply_pending()
                self._handle_lap_line()

                arming = t0 < self.arm_at and self.mode in ("drive", "learn")
                if arming:
                    self.act.release_all()
                    if t0 - last_log > 0.2:
                        print(f"\r старт через {self.arm_at - t0:3.1f}с... ", end="")
                        last_log = t0
                elif self.mode == "record":
                    tel = self.step_record()
                    if t0 - last_log > 0.4:
                        print(f"\r [ЗАПИСЬ] кадров={len(self.demo):5d} "
                              f"err{tel['error']:+.2f} conf{tel['conf']:.2f} v{tel['speed']:.2f}  ",
                              end="")
                        last_log = t0
                elif self.mode == "idle" and self.always_learn:
                    conf, speed, driving = self._observe(t0)
                    if t0 - last_log > 0.8:
                        st = "едешь — учусь" if driving else "жду езды"
                        print(f"\r [фон] {st}: кадров={len(self.bg_buffer):5d} "
                              f"conf={conf:.2f} v{speed:.2f}   ", end="")
                        last_log = t0
                elif self.mode in ("drive", "learn"):
                    learning = self.mode == "learn"
                    tel = self.step_auto(learning)
                    if learning and t0 - self.episode_start >= self.episode_seconds:
                        self._finish_episode()
                    if t0 - last_log > 0.4 and not tel["crash"]:
                        bar = self._bar(self.smooth_error)
                        flags = "ERS" if (self.use_ers and self.straight_frames >=
                                          self.p["ers_straight_frames"]) else "-"
                        ep = ""
                        if learning:
                            left = max(0, self.episode_seconds - (t0 - self.episode_start))
                            ep = f" G{self.brain.generation} t-{left:4.1f}s"
                        print(f"\r {bar} err{tel['error']:+.2f} crv{self.curv:+.2f} "
                              f"conf{tel['conf']:.2f} v{tel['speed']:.2f} "
                              f"[{self.drive_state[:3]}|{flags}] ав:{self.crashes}{ep}  ", end="")
                        last_log = t0

                dt = time.perf_counter() - t0
                if dt < self.target_dt:
                    time.sleep(self.target_dt - dt)
        finally:
            self.act.release_all()
            self._on_exit_record()
            if self.bg_new:
                self.brain.append_demos(self.bg_new)
            self.lap.save()
            print("\nВыход. Мозг и карта трассы сохранены. Кнопки отпущены.")

    def _apply_pending(self):
        """Перехват: если выходим из режима записи — сначала обучить по показу."""
        if self._pending is not None and self.mode == "record":
            self._on_exit_record()
        self._apply_pending_core()

    def _on_exit_record(self):
        if self.mode == "record" and self.demo:
            print(f"\n[показ] Записано {len(self.demo)} кадров. Учусь твоей манере...")
            self.brain.learn_from_demo(self.demo)
            self.demo = []

    def calibrate(self):
        self.vision.refresh_region()
        self._update_center()
        frame = self.vision.grab()
        sc = self.vision.scene(frame)
        print("\n--- КАЛИБРОВКА (что видит зрение) -----------------------")
        src = ("окно Roblox" if self.vision.using_window else "ВЕСЬ ЭКРАН (Roblox не найден!)")
        print(f"  Источник: {src}  {self.vision.region['width']}x{self.vision.region['height']}")
        print(self.vision.debug_ascii(frame))
        rc = self.vision.ref_color
        rc_s = f"BGR({rc[0]:.0f},{rc[1]:.0f},{rc[2]:.0f})" if rc is not None else "—"
        print(f"  '#' = трасса (цвет выучен сам: {rc_s}).  conf={sc['conf']:.2f}")
        print(f"  Дорога впереди: близь={sc['near']:+.2f} средне={sc['mid']:+.2f} "
              f"даль={sc['far']:+.2f} кривизна={sc['curv']:+.2f}")
        print("  Мало '#' -> увеличь vision.color_tol; ловит лишнее -> уменьши.")
        bp = self.brain.best_params
        print(f"  Выучено о езде ({len(self.bg_buffer)} кадров стиля): "
              f"руль={bp['gain']:.0f} тормоз@{bp['brake_error']:.2f} "
              f"сброс@{bp['lift_error']:.2f} апекс={bp['line_weight']:.2f}")
        print(f"  Круг: лучший={self.lap._fmt(self.lap.best_lap)}  "
              f"изучено трассы={self.lap.known_fraction() * 100:.0f}%  "
              f"идеал.линия={self.lap.ideal_known() * 100:.0f}%  "
              f"опасных мест(cut)={len(self.lap.cuts)}  "
              f"линия={'отмечена' if self.lap.anchored else 'нет (жми F4 на старте!)'}")
        print("---------------------------------------------------------")

    @staticmethod
    def _bar(error):
        n = 21
        pos = max(0, min(n - 1, int((error + 1) / 2 * (n - 1))))
        cells = ["-"] * n
        cells[n // 2] = "|"
        cells[pos] = "O"
        return "".join(cells)


def main():
    set_dpi_aware()
    Autopilot(load_cfg()).run()


if __name__ == "__main__":
    main()
