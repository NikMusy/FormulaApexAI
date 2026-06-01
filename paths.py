"""Пути к данным/ресурсам — работают и при запуске из .py, и из собранного .exe."""
import os
import sys


def data_dir() -> str:
    """Куда писать данные (brain/demos/track_map). Рядом с .exe или с исходниками."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(name: str) -> str:
    """Путь к упакованному ресурсу (config.json) внутри .exe или рядом с исходниками."""
    base = getattr(sys, "_MEIPASS", data_dir())
    return os.path.join(base, name)
