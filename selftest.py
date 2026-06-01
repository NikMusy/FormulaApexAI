"""Самопроверка без игры: зрение, ввод, мозг, обучение по показу."""
import os
import time
import random
import numpy as np
from vision import TrackVision
import learning
from autopilot import load_cfg
import controller as C

# не трогаем реальный мозг — пишем в temp
learning.BRAIN_PATH = os.path.join(os.path.dirname(__file__), "_test_brain.json")
learning.DEMO_PATH = os.path.join(os.path.dirname(__file__), "_test_demos.json")
from learning import Brain

cfg = load_cfg()
print("[1] config ок. Руль:", cfg["steering"]["mode"], "| клавиши:", cfg["keys"])

v = TrackVision(cfg)
print("[2] Захват:", v.region["width"], "x", v.region["height"])

f1 = v.grab(); err, conf = v.analyze(f1)
s1 = v.gray_strip(f1); time.sleep(0.03); s2 = v.gray_strip(v.grab())
print(f"[3] Трасса err={err:+.2f} conf={conf:.2f}  скорость={v.speed_proxy(s1,s2,6):.2f}")

print("[4] Привязки:", {k: C._parse(s) for k, s in cfg["keys"].items()})

x, y = C.get_cursor()
print(f"[5] Курсор сейчас ({x},{y}); ЛКМ={C.spec_pressed('mouse:left')} G={C.key_pressed(0x47)}")

print("[6] Обучение по ПОКАЗУ (синтетический заезд):")
b = Brain(cfg)
# имитируем игрока: руль offset = 350*error; тормоз при |err|>0.5; газ когда ровно
demo = []
for _ in range(300):
    e = random.uniform(-0.8, 0.8)
    ae = abs(e)
    demo.append({
        "error": e, "conf": 0.2, "speed": 0.5,
        "offset": 350 * e + random.uniform(-10, 10),
        "throttle": ae < 0.3, "brake": ae > 0.5, "ers": ae < 0.08,
    })
ok = b.learn_from_demo(demo)
print(f"      обучено={ok}  выученный gain={b.best_params['gain']:.0f} "
      f"brake_error={b.best_params['brake_error']:.2f} lift={b.best_params['lift_error']:.2f}")

print("[7] Тест ввода (ЛКМ-клик):")
a = C.Actuator(); a.hold("mouse:left"); time.sleep(0.03); a.release("mouse:left")

for fp in (learning.BRAIN_PATH, learning.DEMO_PATH):
    if os.path.exists(fp):
        os.remove(fp)
print("OK — всё рабочее.")
