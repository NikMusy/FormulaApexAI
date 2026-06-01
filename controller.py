"""
Низкоуровневый ввод в игру через WinAPI SendInput.

Поддерживает и клавиши, и кнопки мыши через единый "спек":
  "mouse:left"   — левая кнопка мыши (газ)
  "mouse:right"  — правая кнопка мыши (тормоз/назад)
  "mouse:middle" — средняя
  "g", "w", "space", ... — клавиша клавиатуры
  ""             — пусто, действие отключено (например, если нет DRS)

Почему SendInput: Roblox читает RAW-ввод мыши и скан-коды клавиатуры, обычные
эмуляторы он часто игнорирует.
"""

import ctypes

_user32 = ctypes.WinDLL("user32", use_last_error=True)
PUL = ctypes.POINTER(ctypes.c_ulong)


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long), ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong), ("dwExtraInfo", PUL),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class _U(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _U)]


INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_EXTENDEDKEY = 0x0001
MOUSEEVENTF_MOVE = 0x0001

_MOUSE_FLAGS = {
    "left":   (0x0002, 0x0004),   # down, up
    "right":  (0x0008, 0x0010),
    "middle": (0x0020, 0x0040),
}

_VK = {
    "lshift": 0xA0, "shift": 0x10, "lctrl": 0xA2, "ctrl": 0x11,
    "space": 0x20, "tab": 0x09, "enter": 0x0D, "alt": 0x12,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
}
for _c in range(ord("a"), ord("z") + 1):
    _VK[chr(_c)] = ord(chr(_c).upper())
for _d in range(10):
    _VK[str(_d)] = ord(str(_d))
_EXTENDED = {0x26, 0x28, 0x25, 0x27}


def _send(inp: INPUT) -> None:
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def move_mouse(dx: int, dy: int = 0) -> None:
    """Относительное смещение мыши — так рулит машина в mouse-steer играх."""
    if dx == 0 and dy == 0:
        return
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
    _send(INPUT(INPUT_MOUSE, _U(mi=mi)))


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


VK_LBUTTON = 0x01
VK_RBUTTON = 0x02


def get_cursor():
    pt = POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def set_cursor(x: int, y: int) -> None:
    """Абсолютная установка курсора — для рулёжки без 'уезжания' (mouse-steer игры)."""
    _user32.SetCursorPos(int(x), int(y))


def key_pressed(vk: int) -> bool:
    """Физическое состояние клавиши/кнопки (работает глобально)."""
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


def spec_pressed(spec: str) -> bool:
    """Нажат ли данный спек (mouse:left/right или клавиша) ПОЛЬЗОВАТЕЛЕМ сейчас."""
    kind, val = _parse(spec)
    if kind == "mouse":
        return key_pressed(VK_LBUTTON if val == "left"
                           else VK_RBUTTON if val == "right" else 0x04)
    if kind == "key":
        return key_pressed(_VK[val])
    return False


def _mouse_button(button: str, up: bool) -> None:
    down_flag, up_flag = _MOUSE_FLAGS[button]
    flag = up_flag if up else down_flag
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(0, 0, 0, flag, 0, ctypes.pointer(extra))
    _send(INPUT(INPUT_MOUSE, _U(mi=mi)))


def _key(name: str, up: bool) -> None:
    vk = _VK[name]
    sc = _user32.MapVirtualKeyW(vk, 0)
    flags = KEYEVENTF_SCANCODE
    if vk in _EXTENDED:
        flags |= KEYEVENTF_EXTENDEDKEY
    if up:
        flags |= KEYEVENTF_KEYUP
    extra = ctypes.c_ulong(0)
    ki = KEYBDINPUT(0, sc, flags, 0, ctypes.pointer(extra))
    _send(INPUT(INPUT_KEYBOARD, _U(ki=ki)))


def _parse(spec: str):
    """Возвращает ('mouse', button) | ('key', name) | (None, None)."""
    if not spec:
        return (None, None)
    spec = spec.strip().lower()
    if spec.startswith("mouse:"):
        b = spec.split(":", 1)[1]
        if b not in _MOUSE_FLAGS:
            raise ValueError(f"Неизвестная кнопка мыши: {spec!r}")
        return ("mouse", b)
    if spec not in _VK:
        raise ValueError(f"Неизвестная клавиша в config.json: {spec!r}")
    return ("key", spec)


class Actuator:
    """Держит учёт зажатых клавиш/кнопок, чтобы не дёргать их каждый кадр."""

    def __init__(self):
        self._down = set()

    def _act(self, spec: str, up: bool) -> None:
        kind, val = _parse(spec)
        if kind == "mouse":
            _mouse_button(val, up)
        elif kind == "key":
            _key(val, up)

    def hold(self, spec: str) -> None:
        if spec and spec not in self._down:
            self._act(spec, up=False)
            self._down.add(spec)

    def release(self, spec: str) -> None:
        if spec and spec in self._down:
            self._act(spec, up=True)
            self._down.discard(spec)

    def set(self, spec: str, pressed: bool) -> None:
        self.hold(spec) if pressed else self.release(spec)

    def release_all(self) -> None:
        for spec in list(self._down):
            self.release(spec)
