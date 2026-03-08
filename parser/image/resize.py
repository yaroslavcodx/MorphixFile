"""
Функции изменения размера изображений (resize / downscale).

Поддерживает:
- Уменьшение (downscale)
- Увеличение (upscale через resize)
- Различные алгоритмы ресемплинга
- Пакетную обработку

Ограничения:
- Максимальный размер: 16384x16384 пикселей
- Максимальный scale: 16x
- Минимальный размер: 1x1 пиксель
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from PIL import Image

from parser.image.utils import (
    ImageStats,
    load_image,
    save_image,
    calculate_dimensions,
    get_resampling_method,
    SUPPORTED_FORMATS,
)

# Константы ограничений
MAX_DIMENSION = 16384  # Максимальный размер по одной стороне
MAX_SCALE = 16.0       # Максимальный коэффициент увеличения
MIN_SCALE = 0.01       # Минимальный коэффициент уменьшения
MIN_DIMENSION = 1      # Минимальный размер


def resize_image(
    input_file: str,
    output_file: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[Union[int, float]] = None,
    method: str = "lanczos",
    quality: int = 95,
    return_stats: bool = False,
) -> Union[str, Tuple[str, ImageStats]]:
    """
    Изменение размера изображения.

    Args:
        input_file: Путь к входному файлу.
        output_file: Путь к выходному файлу (автогенерация если None).
        width: Целевая ширина в пикселях.
        height: Целевая высота в пикселях.
        scale: Коэффициент масштабирования (0.5 = 50%, 2 = 200%). Диапазон: 0.01-16.
        method: Алгоритм ресемплинга (nearest, bilinear, bicubic, lanczos).
        quality: Качество сохранения (1-100) для JPEG/WebP.
        return_stats: Вернуть статистику вместе с путем.

    Returns:
        Путь к сохраненному файлу или кортеж (путь, статистика).

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если формат не поддерживается или параметры некорректны.
    """
    stats = ImageStats()
    stats.start(input_file)
    stats.method = f"Resize ({method})"

    try:
        # Загрузка изображения
        img = load_image(input_file)
        stats.original_size = img.size
        orig_width, orig_height = img.size

        # Конвертация и валидация scale
        if scale is not None:
            # Если scale - целое число > 1 и <= 100, считаем что это проценты
            if isinstance(scale, int) and scale > 1 and scale <= 100:
                scale = scale / 100.0
            # Float: если > 10 и <= 100, считаем что это проценты
            elif isinstance(scale, float) and scale > 10 and scale <= 100:
                scale = scale / 100.0
            
            # Ограничение диапазона scale
            if scale < MIN_SCALE:
                scale = MIN_SCALE
                stats.filters_applied.append(f"Scale clamped to min: {MIN_SCALE}")
            elif scale > MAX_SCALE:
                scale = MAX_SCALE
                stats.filters_applied.append(f"Scale clamped to max: {MAX_SCALE}")

        # Валидация width/height
        if width is not None:
            if width < MIN_DIMENSION:
                raise ValueError(f"Ширина должна быть >= {MIN_DIMENSION}")
            if width > MAX_DIMENSION:
                raise ValueError(f"Ширина превышает максимум ({MAX_DIMENSION})")
        
        if height is not None:
            if height < MIN_DIMENSION:
                raise ValueError(f"Высота должна быть >= {MIN_DIMENSION}")
            if height > MAX_DIMENSION:
                raise ValueError(f"Высота превышает максимум ({MAX_DIMENSION})")

        # Расчет новых размеров
        new_width, new_height = calculate_dimensions(
            img, width=width, height=height, scale=scale
        )

        # Финальная проверка максимального размера
        if new_width > MAX_DIMENSION or new_height > MAX_DIMENSION:
            ratio = min(MAX_DIMENSION / new_width, MAX_DIMENSION / new_height)
            new_width = int(new_width * ratio)
            new_height = int(new_height * ratio)
            stats.filters_applied.append(f"Clamped to max: {new_width}x{new_height}")

        # Проверка минимального размера
        if new_width < MIN_DIMENSION or new_height < MIN_DIMENSION:
            raise ValueError(
                f"Размер слишком мал: {new_width}x{new_height}. "
                f"Минимум: {MIN_DIMENSION}x{MIN_DIMENSION}"
            )

        # Получение метода ресемплинга
        resampling = get_resampling_method(method)

        # Изменение размера
        resized_img = img.resize((new_width, new_height), resample=resampling)
        stats.result_size = resized_img.size

        # Определение выходного файла
        if output_file is None:
            input_path = Path(input_file)
            suffix = f"_resized_{new_width}x{new_height}"
            output_file = str(input_path.with_stem(input_path.stem + suffix))

        # Сохранение
        save_image(resized_img, output_file, quality=quality)
        stats.result_file_size = Path(output_file).stat().st_size

        stats.end()

        if return_stats:
            return output_file, stats
        return output_file

    except Exception as e:
        stats.add_error(str(e))
        stats.end()
        raise


def downscale_image(
    input_file: str,
    output_file: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[Union[int, float]] = None,
    method: str = "lanczos",
    quality: int = 95,
    return_stats: bool = False,
) -> Union[str, Tuple[str, ImageStats]]:
    """
    Уменьшение изображения (downscale).

    Алиас для resize_image с валидацией на уменьшение.

    Args:
        input_file: Путь к входному файлу.
        output_file: Путь к выходному файлу.
        width: Целевая ширина.
        height: Целевая высота.
        scale: Коэффициент уменьшения (0.5 = 50%).
        method: Алгоритм ресемплинга.
        quality: Качество сохранения.
        return_stats: Вернуть статистику.

    Returns:
        Путь к файлу или кортеж (путь, статистика).
    """
    stats = ImageStats()
    stats.start(input_file)

    try:
        img = load_image(input_file)
        orig_width, orig_height = img.size

        # Расчет целевых размеров
        target_width, target_height = calculate_dimensions(
            img, width=width, height=height, scale=scale
        )

        # Проверка что это действительно уменьшение
        if target_width > orig_width or target_height > orig_height:
            raise ValueError(
                "downscale_image предназначен только для уменьшения. "
                "Для увеличения используйте upscale_image."
            )

        stats.end()

        # Вызов resize_image
        return resize_image(
            input_file,
            output_file=output_file,
            width=width,
            height=height,
            scale=scale,
            method=method,
            quality=quality,
            return_stats=return_stats,
        )

    except Exception as e:
        stats.add_error(str(e))
        stats.end()
        raise


def resize_batch(
    input_files: list,
    output_dir: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[Union[int, float]] = None,
    method: str = "lanczos",
    quality: int = 95,
) -> Dict[str, Any]:
    """
    Пакетное изменение размера изображений.

    Args:
        input_files: Список путей к файлам.
        output_dir: Директория для сохранения.
        width: Целевая ширина.
        height: Целевая высота.
        scale: Коэффициент масштабирования.
        method: Алгоритм ресемплинга.
        quality: Качество сохранения.

    Returns:
        Словарь с результатами обработки.
    """
    results = {}

    for file_path in input_files:
        try:
            if output_dir:
                input_path = Path(file_path)
                output_file = str(Path(output_dir) / input_path.name)
            else:
                output_file = None

            output_path, stats = resize_image(
                file_path,
                output_file=output_file,
                width=width,
                height=height,
                scale=scale,
                method=method,
                quality=quality,
                return_stats=True,
            )

            results[file_path] = {
                "success": True,
                "output": output_path,
                "stats": stats.to_dict(),
            }

        except Exception as e:
            results[file_path] = {
                "success": False,
                "error": str(e),
            }

    return results
