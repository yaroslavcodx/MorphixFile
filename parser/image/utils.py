"""
Утилиты для обработки изображений.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime
from PIL import Image

# Поддерживаемые форматы изображений
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"]

# Маппинг расширений к форматам PIL
EXTENSION_TO_FORMAT = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tiff": "TIFF",
    ".tif": "TIFF",
}

# Алгоритмы ресемплинга
RESAMPLING_METHODS = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "box": Image.BOX,
    "hamming": Image.HAMMING,
}


class ImageStats:
    """Статистика обработки изображения."""

    def __init__(self):
        self.file_path: str = ""
        self.original_size: Tuple[int, int] = (0, 0)
        self.result_size: Tuple[int, int] = (0, 0)
        self.original_file_size: int = 0
        self.result_file_size: int = 0
        self.method: str = ""
        self.filters_applied: List[str] = []
        self.processing_time: float = 0.0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.errors: List[str] = []

    def start(self, file_path: str):
        """Начало обработки."""
        self.start_time = datetime.now()
        self.file_path = file_path
        self.original_file_size = Path(file_path).stat().st_size

    def end(self):
        """Завершение обработки."""
        self.end_time = datetime.now()
        if self.start_time:
            self.processing_time = (self.end_time - self.start_time).total_seconds()

    def add_error(self, error: str):
        """Добавление ошибки."""
        self.errors.append(error)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "file_path": self.file_path,
            "original_size": f"{self.original_size[0]}x{self.original_size[1]}",
            "result_size": f"{self.result_size[0]}x{self.result_size[1]}",
            "original_file_size": _format_size(self.original_file_size),
            "result_file_size": _format_size(self.result_file_size),
            "method": self.method,
            "filters_applied": self.filters_applied,
            "processing_time": f"{self.processing_time:.3f}s",
            "errors": self.errors,
        }

    def __str__(self) -> str:
        """Строковое представление."""
        lines = [
            f"Файл: {self.file_path}",
            f"Исходный размер: {self.original_size[0]}x{self.original_size[1]}",
            f"Новый размер: {self.result_size[0]}x{self.result_size[1]}",
            f"Метод: {self.method}",
        ]
        if self.filters_applied:
            lines.append(f"Фильтры: {', '.join(self.filters_applied)}")
        lines.append(f"Время обработки: {self.processing_time:.3f}s")
        if self.errors:
            lines.append("Ошибки:")
            for error in self.errors:
                lines.append(f"  - {error}")
        return "\n".join(lines)


def _format_size(size_bytes: int) -> str:
    """Форматирование размера файла."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def get_image_info(file_path: str) -> Dict[str, Any]:
    """
    Получение информации об изображении.

    Args:
        file_path: Путь к изображению.

    Returns:
        Словарь с информацией об изображении.

    Raises:
        ValueError: Если файл не является изображением.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    with Image.open(path) as img:
        return {
            "file": str(path),
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "width": img.width,
            "height": img.height,
            "file_size": path.stat().st_size,
            "file_size_formatted": _format_size(path.stat().st_size),
        }


def load_image(file_path: str) -> Image.Image:
    """
    Загрузка изображения.

    Args:
        file_path: Путь к изображению.

    Returns:
        PIL Image объект.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если формат не поддерживается.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Неподдерживаемый формат: {ext}. "
            f"Поддерживаемые: {', '.join(SUPPORTED_FORMATS)}"
        )

    img = Image.open(path)
    # Конвертируем в RGB для PNG с прозрачностью
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


def save_image(
    img: Image.Image,
    file_path: str,
    quality: int = 95,
    format_name: Optional[str] = None,
) -> str:
    """
    Сохранение изображения.

    Args:
        img: PIL Image объект.
        file_path: Путь для сохранения.
        quality: Качество для JPEG/WebP (1-100).
        format_name: Формат файла (автоопределение по расширению).

    Returns:
        Путь к сохраненному файлу.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()

    if format_name is None:
        format_name = EXTENSION_TO_FORMAT.get(ext, "JPEG")

    # Параметры сохранения в зависимости от формата
    save_kwargs = {}
    if format_name in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    elif format_name == "PNG":
        save_kwargs["optimize"] = True

    img.save(path, format=format_name, **save_kwargs)

    return str(path)


def calculate_dimensions(
    img: Image.Image,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[float] = None,
) -> Tuple[int, int]:
    """
    Расчет новых размеров изображения.

    Args:
        img: PIL Image объект.
        width: Целевая ширина.
        height: Целевая высота.
        scale: Коэффициент масштабирования (0.5 = 50%, 2.0 = 200%).

    Returns:
        Кортеж (width, height).

    Raises:
        ValueError: Если не указаны параметры.
    """
    orig_width, orig_height = img.size

    if scale is not None:
        # Масштабирование по проценту
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
    elif width is not None and height is not None:
        # Заданы оба размера
        new_width = width
        new_height = height
    elif width is not None:
        # Только ширина с сохранением пропорций
        aspect_ratio = orig_height / orig_width
        new_width = width
        new_height = int(width * aspect_ratio)
    elif height is not None:
        # Только высота с сохранением пропорций
        aspect_ratio = orig_width / orig_height
        new_height = height
        new_width = int(height * aspect_ratio)
    else:
        raise ValueError("Необходимо указать width, height или scale")

    # Проверка на валидность
    if new_width <= 0 or new_height <= 0:
        raise ValueError("Размеры должны быть положительными")

    return new_width, new_height


def get_resampling_method(method_name: str) -> int:
    """
    Получение константы ресемплинга по названию.

    Args:
        method_name: Название метода.

    Returns:
        Константа PIL.
    """
    return RESAMPLING_METHODS.get(method_name.lower(), Image.LANCZOS)


def is_supported_image(file_path: str) -> bool:
    """
    Проверка поддержки формата изображения.

    Args:
        file_path: Путь к файлу.

    Returns:
        True если формат поддерживается.
    """
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_FORMATS


def find_images(
    folder: str,
    recursive: bool = False,
    extensions: Optional[List[str]] = None,
) -> List[str]:
    """
    Поиск изображений в папке.

    Args:
        folder: Путь к папке.
        recursive: Рекурсивный поиск.
        extensions: Список расширений для поиска.

    Returns:
        Список путей к изображениям.
    """
    if extensions is None:
        extensions = SUPPORTED_FORMATS

    folder_path = Path(folder)
    images = []

    if recursive:
        for ext in extensions:
            images.extend(folder_path.rglob(f"*{ext}"))
            images.extend(folder_path.rglob(f"*{ext.upper()}"))
    else:
        for ext in extensions:
            images.extend(folder_path.glob(f"*{ext}"))
            images.extend(folder_path.glob(f"*{ext.upper()}"))

    return [str(img) for img in images]
