"""
Функции улучшения качества изображений (enhance).

Поддерживает:
- Повышение резкости (sharpening)
- Подавление шумов (noise reduction)
- Улучшение контраста (contrast enhancement)
- Авто-коррекция яркости (auto brightness)
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from PIL import Image, ImageFilter, ImageEnhance

from parser.image.utils import (
    ImageStats,
    load_image,
    save_image,
    SUPPORTED_FORMATS,
)


def apply_sharpening(
    img: Image.Image,
    factor: float = 1.5,
    radius: int = 2,
) -> Image.Image:
    """
    Применение фильтра резкости.

    Args:
        img: PIL Image объект.
        factor: Коэффициент резкости (1.0 = без эффекта, >1 = усиление).
        radius: Радиус фильтра.

    Returns:
        Обработанное изображение.
    """
    # Базовый фильтр резкости
    sharpen_filter = ImageFilter.UnsharpMask(radius=radius, percent=int(factor * 100), threshold=3)
    return img.filter(sharpen_filter)


def apply_noise_reduction(
    img: Image.Image,
    radius: int = 2,
) -> Image.Image:
    """
    Применение шумоподавления.

    Args:
        img: PIL Image объект.
        radius: Радиус фильтра (большее = сильнее сглаживание).

    Returns:
        Обработанное изображение.
    """
    # Используем Gaussian Blur с небольшим радиусом
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_contrast_enhancement(
    img: Image.Image,
    factor: float = 1.2,
) -> Image.Image:
    """
    Улучшение контраста.

    Args:
        img: PIL Image объект.
        factor: Коэффициент контраста (1.0 = без эффекта, >1 = усиление).

    Returns:
        Обработанное изображение.
    """
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def apply_brightness_correction(
    img: Image.Image,
    factor: Optional[float] = None,
    auto: bool = False,
) -> Image.Image:
    """
    Коррекция яркости.

    Args:
        img: PIL Image объект.
        factor: Коэффициент яркости (1.0 = без эффекта, >1 = светлее).
        auto: Автоматическая коррекция на основе гистограммы.

    Returns:
        Обработанное изображение.
    """
    if auto:
        # Автоматическая коррекция на основе средней яркости
        grayscale = img.convert("L")
        histogram = grayscale.histogram()
        
        # Вычисление средней яркости
        total_pixels = sum(histogram)
        mean_brightness = sum(i * count for i, count in enumerate(histogram)) / total_pixels
        
        # Расчет коэффициента для достижения средней яркости ~128
        target_brightness = 128
        factor = target_brightness / mean_brightness if mean_brightness > 0 else 1.0
        factor = max(0.5, min(2.0, factor))  # Ограничение
    
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_color_enhancement(
    img: Image.Image,
    factor: float = 1.1,
) -> Image.Image:
    """
    Улучшение насыщенности цвета.

    Args:
        img: PIL Image объект.
        factor: Коэффициент насыщенности (1.0 = без эффекта, >1 = усиление).

    Returns:
        Обработанное изображение.
    """
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)


def enhance_image(
    input_file: str,
    output_file: Optional[str] = None,
    sharpen: bool = True,
    sharpen_factor: float = 1.5,
    noise_reduction: bool = False,
    noise_radius: int = 1,
    contrast: bool = True,
    contrast_factor: float = 1.1,
    brightness: bool = False,
    brightness_factor: Optional[float] = None,
    auto_brightness: bool = True,
    color: bool = False,
    color_factor: float = 1.1,
    quality: int = 95,
    return_stats: bool = False,
) -> Union[str, Tuple[str, ImageStats]]:
    """
    Комплексное улучшение качества изображения.

    Args:
        input_file: Путь к входному файлу.
        output_file: Путь к выходному файлу (автогенерация если None).
        sharpen: Применить повышение резкости.
        sharpen_factor: Коэффициент резкости.
        noise_reduction: Применить шумоподавление.
        noise_radius: Радиус шумоподавления.
        contrast: Применить улучшение контраста.
        contrast_factor: Коэффициент контраста.
        brightness: Применить коррекцию яркости.
        brightness_factor: Коэффициент яркости.
        auto_brightness: Автоматическая коррекция яркости.
        color: Применить улучшение цвета.
        color_factor: Коэффициент насыщенности.
        quality: Качество сохранения (1-100).
        return_stats: Вернуть статистику вместе с путем.

    Returns:
        Путь к сохраненному файлу или кортеж (путь, статистика).
    """
    stats = ImageStats()
    stats.start(input_file)
    stats.method = "Enhance"

    try:
        # Загрузка изображения
        img = load_image(input_file)
        stats.original_size = img.size

        result_img = img

        # Применение фильтров в порядке
        if noise_reduction:
            result_img = apply_noise_reduction(result_img, radius=noise_radius)
            stats.filters_applied.append(f"Noise Reduction (r={noise_radius})")

        if sharpen:
            result_img = apply_sharpening(result_img, factor=sharpen_factor)
            stats.filters_applied.append(f"Sharpen (f={sharpen_factor})")

        if contrast:
            result_img = apply_contrast_enhancement(result_img, factor=contrast_factor)
            stats.filters_applied.append(f"Contrast (f={contrast_factor})")

        if brightness or auto_brightness:
            result_img = apply_brightness_correction(
                result_img,
                factor=brightness_factor,
                auto=auto_brightness,
            )
            if auto_brightness:
                stats.filters_applied.append("Auto Brightness")
            else:
                stats.filters_applied.append(f"Brightness (f={brightness_factor})")

        if color:
            result_img = apply_color_enhancement(result_img, factor=color_factor)
            stats.filters_applied.append(f"Color (f={color_factor})")

        stats.result_size = result_img.size

        # Определение выходного файла
        if output_file is None:
            input_path = Path(input_file)
            suffix = "_enhanced"
            output_file = str(input_path.with_stem(input_path.stem + suffix))

        # Сохранение
        save_image(result_img, output_file, quality=quality)
        stats.result_file_size = Path(output_file).stat().st_size

        stats.end()

        if return_stats:
            return output_file, stats
        return output_file

    except Exception as e:
        stats.add_error(str(e))
        stats.end()
        raise


def quick_enhance(
    input_file: str,
    output_file: Optional[str] = None,
    quality: int = 95,
    return_stats: bool = False,
) -> Union[str, Tuple[str, ImageStats]]:
    """
    Быстрое улучшение изображения с предустановленными параметрами.

    Применяет:
    - Легкое повышение резкости (factor=1.3)
    - Легкое улучшение контраста (factor=1.1)
    - Авто-коррекция яркости

    Args:
        input_file: Путь к входному файлу.
        output_file: Путь к выходному файлу.
        quality: Качество сохранения.
        return_stats: Вернуть статистику.

    Returns:
        Путь к файлу или кортет (путь, статистика).
    """
    return enhance_image(
        input_file,
        output_file=output_file,
        sharpen=True,
        sharpen_factor=1.3,
        noise_reduction=False,
        contrast=True,
        contrast_factor=1.1,
        brightness=False,
        auto_brightness=True,
        color=False,
        quality=quality,
        return_stats=return_stats,
    )


def enhance_batch(
    input_files: List[str],
    output_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Пакетное улучшение изображений.

    Args:
        input_files: Список путей к файлам.
        output_dir: Директория для сохранения.
        **kwargs: Параметры для enhance_image.

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

            output_path, stats = enhance_image(
                file_path,
                output_file=output_file,
                return_stats=True,
                **kwargs,
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
