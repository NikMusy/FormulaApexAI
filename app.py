"""
Formula Apex AI — графическое приложение (панель управления).

Запускает автопилот в фоновом потоке и показывает живую телеметрию:
режим, уверенность зрения, скорость, кривизна, время круга, % изученной
трассы, сколько кадров твоей езды в памяти. Кнопки дублируют горячие клавиши.

Запуск: Запустить-GUI.bat (от админа). Горячие клавиши тоже работают.
"""

import threading
import tkinter as tk
from tkinter import font as tkfont

from window import set_dpi_aware
from autopilot import Autopilot, load_cfg

BG = "#0d1117"
FG = "#e6edf3"
ACCENT = "#f78166"
GREEN = "#3fb950"
MUTED = "#8b949e"
CARD = "#161b22"


class App:
    def __init__(self):
        set_dpi_aware()
        self.ap = Autopilot(load_cfg())
        self.thread = threading.Thread(target=self.ap.run, daemon=True)
        self.thread.start()

        self.root = tk.Tk()
        self.root.title("Formula Apex AI 🏎️")
        self.root.configure(bg=BG)
        self.root.geometry("440x660")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.big = tkfont.Font(family="Consolas", size=13, weight="bold")
        self.mid = tkfont.Font(family="Consolas", size=11)
        self.small = tkfont.Font(family="Consolas", size=9)

        tk.Label(self.root, text="FORMULA APEX AI", bg=BG, fg=ACCENT,
                 font=tkfont.Font(family="Consolas", size=18, weight="bold")).pack(pady=(14, 2))
        self.mode_lbl = tk.Label(self.root, text="режим: —", bg=BG, fg=GREEN, font=self.big)
        self.mode_lbl.pack(pady=(0, 8))

        # карточки телеметрии
        grid = tk.Frame(self.root, bg=BG)
        grid.pack(padx=14, fill="x")
        self.cells = {}
        specs = [("conf", "Зрение"), ("speed", "Скорость"), ("curv", "Кривизна"),
                 ("state", "Газ/тормоз"), ("best", "Лучший круг"), ("last", "Прошлый круг"),
                 ("learned", "Трасса изучена"), ("ideal", "Идеал. линия"),
                 ("frames", "Кадров в памяти"), ("cuts", "Опасных мест"),
                 ("crashes", "Аварий"), ("mode2", "Линия")]
        for idx, (key, title) in enumerate(specs):
            r, c = divmod(idx, 2)
            card = tk.Frame(grid, bg=CARD, padx=10, pady=7)
            card.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            grid.grid_columnconfigure(c, weight=1)
            tk.Label(card, text=title, bg=CARD, fg=MUTED, font=self.small).pack(anchor="w")
            v = tk.Label(card, text="—", bg=CARD, fg=FG, font=self.mid)
            v.pack(anchor="w")
            self.cells[key] = v

        # кнопки
        btns = tk.Frame(self.root, bg=BG)
        btns.pack(pady=12, padx=14, fill="x")
        self._btn(btns, "● ЗАПИСЬ (F5)", lambda: self.ap.request("record"), 0, 0)
        self._btn(btns, "🧠 ОБУЧЕНИЕ (F6)", lambda: self.ap.request("learn"), 0, 1)
        self._btn(btns, "🚗 ЕХАТЬ (F8)", lambda: self.ap.request("drive"), 1, 0, GREEN)
        self._btn(btns, "■ СТОП", lambda: self.ap.request("idle"), 1, 1)
        self._btn(btns, "🏁 КРУГ старт/финиш (F4)", self.ap.request_lap_line, 2, 0)
        self._btn(btns, "🔍 КАЛИБРОВКА (F7)", self.ap.calibrate, 2, 1)
        self._btn(btns, "🧠 AI-РАЗБОР ЗАЕЗДА (F2)", self.ap.request_coach, 3, 0, ACCENT, colspan=2)
        for c in (0, 1):
            btns.grid_columnconfigure(c, weight=1)

        self.hint = tk.Label(self.root, bg=BG, fg=MUTED, font=self.small, justify="left",
                             text="Открой Roblox → F4 на старте круга → катай сам (учусь)\n"
                                  "→ потом F8, и я поеду как ты. Подробности в консоли.")
        self.hint.pack(pady=(2, 6))

        self.update_loop()

    def _btn(self, parent, text, cmd, r, c, fg=FG, colspan=1):
        b = tk.Button(parent, text=text, command=cmd, bg=CARD, fg=fg,
                      activebackground=ACCENT, activeforeground=BG, font=self.mid,
                      relief="flat", padx=6, pady=8, cursor="hand2")
        b.grid(row=r, column=c, columnspan=colspan, sticky="nsew", padx=4, pady=4)

    def update_loop(self):
        ap = self.ap
        names = {"idle": "ОЖИДАНИЕ (учусь у тебя)", "drive": "ЕДУ САМ",
                 "learn": "ОБУЧЕНИЕ", "record": "ЗАПИСЬ ТВОЕЙ ЕЗДЫ"}
        self.mode_lbl.config(text="режим: " + names.get(ap.mode, ap.mode))
        self.cells["conf"].config(text=f"{ap.last_conf * 100:.0f}%")
        self.cells["speed"].config(text=f"{ap.avg_speed:.2f}")
        self.cells["curv"].config(text=f"{ap.curv:+.2f}")
        self.cells["state"].config(text={"accel": "ГАЗ", "coast": "сброс",
                                          "brake": "ТОРМОЗ"}.get(ap.drive_state, ap.drive_state))
        self.cells["best"].config(text=ap.lap._fmt(ap.lap.best_lap))
        self.cells["last"].config(text=ap.lap._fmt(ap.lap.last_lap))
        self.cells["learned"].config(text=f"{ap.lap.known_fraction() * 100:.0f}%")
        self.cells["ideal"].config(text=f"{ap.lap.ideal_known() * 100:.0f}%")
        self.cells["frames"].config(text=f"{len(ap.bg_buffer)}")
        self.cells["cuts"].config(text=f"{len(ap.lap.cuts)}")
        self.cells["crashes"].config(text=f"{ap.crashes}")
        self.cells["mode2"].config(text="ИДЕАЛ" if ap.ideal_off is not None
                                   else ("твоя" if ap.map_human_off is not None else "зрение"))
        if ap.coach_status:
            self.hint.config(text="Claude: " + ap.coach_status)
        self.root.after(250, self.update_loop)

    def on_close(self):
        self.ap.stop_flag = True
        self.root.after(400, self.root.destroy)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
