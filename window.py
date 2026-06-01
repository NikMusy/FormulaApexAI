"""
Поиск окна Roblox и его области на экране — чтобы захватывать ТОЛЬКО игру,
а не весь рабочий стол. Так зрение чистое и ничего лишнего не палится.
"""

import ctypes
from ctypes import wintypes

_user32 = ctypes.WinDLL("user32", use_last_error=True)


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


_WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)


def set_dpi_aware():
    """Чтобы координаты окна совпадали с реальными пикселями захвата."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PER_MONITOR_AWARE
    except Exception:
        try:
            _user32.SetProcessDPIAware()
        except Exception:
            pass


def _title_of(hwnd):
    n = _user32.GetWindowTextLengthW(hwnd)
    if n == 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    _user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def find_window(title_substr="Roblox"):
    """Возвращает hwnd видимого окна, чей заголовок содержит подстроку, иначе None."""
    title_substr = title_substr.lower()
    matches = []

    def cb(hwnd, _):
        if not _user32.IsWindowVisible(hwnd):
            return True
        t = _title_of(hwnd)
        if t and title_substr in t.lower():
            r = RECT()
            _user32.GetClientRect(hwnd, ctypes.byref(r))
            area = (r.right - r.left) * (r.bottom - r.top)
            matches.append((area, hwnd, t))
        return True

    _user32.EnumWindows(_WNDENUMPROC(cb), 0)
    if not matches:
        return None
    matches.sort(reverse=True)        # самое большое окно
    return matches[0][1]


def window_region(title_substr="Roblox"):
    """
    Область клиентской части окна Roblox в экранных координатах:
    {left, top, width, height} — готово для mss. None, если окна нет.
    """
    hwnd = find_window(title_substr)
    if not hwnd:
        return None
    r = RECT()
    _user32.GetClientRect(hwnd, ctypes.byref(r))
    w, h = r.right - r.left, r.bottom - r.top
    if w <= 0 or h <= 0:
        return None
    p = POINT(0, 0)
    _user32.ClientToScreen(hwnd, ctypes.byref(p))
    return {"left": int(p.x), "top": int(p.y), "width": int(w), "height": int(h),
            "title": _title_of(hwnd)}
