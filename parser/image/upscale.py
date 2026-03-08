"""
Функции увеличения изображений (upscale).

Поддерживает:
- Увеличение по scale, width, height
- Выбор алгоритма ресемплинга
- Продвинутое улучшение качества:
  * Пошаговое увеличение (step-up scaling)
  * Multi-pass sharpening
  * Улучшение детализации краёв
  * Контраст и насыщенность
- Sharpening после увеличения
- Noise reduction
- Edge enhancement

Ограничения:
- Максимальный размер: 16384x16384 пикселей
- Максимальный scale: 16x
- Минимальный scale: 1.01x
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from PIL import Image, ImageFilter, ImageEnhance

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
MIN_SCALE = 1.01       # Минимальный коэффициент увеличения (чтобы было увеличение)
MAX_OUTPUT_FILE_SIZE = 500 * 1024 * 1024  # 500 MB макс. размер файла


def _apply_smart_sharpen(img: Image.Image, factor: float = 1.5) -> Image.Image:
    """
    Умное повышение резкости с защитой от артефактов.
    
    Использует комбинацию фильтров для лучшего результата:
    1. Базовый UnsharpMask
    2. Детализация через Detail фильтр
    """
    # Первый проход - мягкий sharpening
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(factor * 100), threshold=3))
    
    # Второй проход - акцент на деталях
    if factor >= 2.0:
        img = img.filter(ImageFilter.DETAIL)
    
    return img


def _apply_detail_enhancement(img: Image.Image, strength: float = 1.2) -> Image.Image:
    """
    Улучшение детализации через комбинацию фильтров.
    
    Args:
        img: PIL Image объект.
        strength: Коэффициент усиления (1.0-2.0).
    """
    # Улучшение краёв
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    # зашита от излишнего шума
    img = img.filter(ImageFilter.SMOOTH_MORE)
    
    return img


def _step_upscale(img: Image.Image, target_size: Tuple[int, int], method: int) -> Image.Image:
    """
    Пошаговое увеличение изображения для лучшего качества.
    
    Вместо одного большого увеличения делает несколько шагов по 2x,
    что даёт более чёткий результат.
    
    Args:
        img: Исходное изображение.
        target_size: Целевой размер (width, height).
        method: Метод ресемплинга.
    
    Returns:
        Увеличенное изображение.
    """
    current_width, current_height = img.size
    target_width, target_height = target_size
    
    result_img = img
    
    # Увеличиваем поэтапно (максимум 2x за шаг)
    while current_width < target_width * 0.95 or current_height < target_height * 0.95:
        # Вычисляем следующий шаг
        next_width = min(int(current_width * 2), target_width)
        next_height = min(int(current_height * 2), target_height)
        
        # Увеличиваем на этом шаге
        result_img = result_img.resize((next_width, next_height), resample=method)
        
        # Применяем лёгкий sharpening после каждого шага
        result_img = result_img.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=2))
        
        current_width, current_height = next_width, next_height
    
    # Финальное точное увеличение до целевого размера
    if (current_width, current_height) != target_size:
        result_img = result_img.resize(target_size, resample=method)
    
    return result_img


def _enhance_colors(img: Image.Image, contrast: float = 1.1, saturation: float = 1.1, 
                    brightness: float = 1.0) -> Image.Image:
    """
    Улучшение цветовых характеристик.
    
    Args:
        img: PIL Image объект.
        contrast: Коэффициент контраста (1.0 = без изменений).
        saturation: Коэффициент насыщенности.
        brightness: Коэффициент яркости.
    """
    result = img
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(contrast)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(saturation)
    
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(brightness)
    
    return result


def upscale_image(
    input_file: str,
    output_file: Optional[str] = None,
    scale: float = 2.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    method: str = "lanczos",
    sharpen: bool = True,
    sharpen_factor: float = 2.0,
    noise_reduction: bool = False,
    edge_enhance: bool = False,
    enhance_colors: bool = True,
    step_upscale: bool = True,
    quality: int = 95,
    return_stats: bool = False,
    allow_upscale: bool = True,
) -> Union[str, Tuple[str, ImageStats]]:
    """
    Увеличение изображения (upscale) с улучшением качества.

    Args:
        input_file: Путь к входному файлу.
        output_file: Путь к выходному файлу (автогенерация если None).
        scale: Коэффициент увеличения (2 = 200%, 4 = 400%). Диапазон: 1.01-16.
        width: Целевая ширина (переопределяет scale).
        height: Целевая высота (переопределяет scale).
        method: Алгоритм ресемплинга (nearest, bilinear, bicubic, lanczos).
        sharpen: Применить sharpening после увеличения.
        sharpen_factor: Коэффициент резкости (1.0-3.0, по умолчанию 2.0).
        noise_reduction: Применить шумоподавление.
        edge_enhance: Применить улучшение краев.
        enhance_colors: Улучшить контраст и насыщенность.
        step_upscale: Использовать пошаговое увеличение (лучше качество).
        quality: Качество сохранения (1-100).
        return_stats: Вернуть статистику вместе с путем.
        allow_upscale: Разрешить увеличение больше оригинала.

    Returns:
        Путь к сохраненному файлу или кортеж (путь, статистика).

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если формат не поддерживается или параметры некорректны.
    """
    stats = ImageStats()
    stats.start(input_file)
    stats.method = f"Upscale ({method})"

    try:
        # Загрузка изображения
        img = load_image(input_file)
        stats.original_size = img.size
        orig_width, orig_height = img.size

        # Валидация и нормализация scale
        if width is None and height is None:
            # Ограничение scale диапазоном
            if scale < MIN_SCALE:
                scale = MIN_SCALE
                stats.filters_applied.append(f"Scale adjusted to min: {MIN_SCALE}")
            elif scale > MAX_SCALE:
                scale = MAX_SCALE
                stats.filters_applied.append(f"Scale adjusted to max: {MAX_SCALE}")

            scale_factor = scale
        else:
            scale_factor = None

            # Валидация width/height
            if width is not None and width > MAX_DIMENSION:
                raise ValueError(f"Ширина {width} превышает максимум ({MAX_DIMENSION})")
            if height is not None and height > MAX_DIMENSION:
                raise ValueError(f"Высота {height} превышает максимум ({MAX_DIMENSION})")

        # Расчет новых размеров
        new_width, new_height = calculate_dimensions(
            img, width=width, height=height, scale=scale_factor
        )

        # Проверка максимального размера
        if new_width > MAX_DIMENSION or new_height > MAX_DIMENSION:
            # Масштабирование до максимума с сохранением пропорций
            ratio = min(MAX_DIMENSION / new_width, MAX_DIMENSION / new_height)
            new_width = int(new_width * ratio)
            new_height = int(new_height * ratio)
            stats.filters_applied.append(f"Resized to max: {new_width}x{new_height}")

        # Проверка что это действительно upscale (если не заданы width/height явно)
        if allow_upscale is False and (new_width > orig_width or new_height > orig_height):
            raise ValueError(
                f"Upscale запрещён: {orig_width}x{orig_height} -> {new_width}x{new_height}. "
                "Используйте allow_upscale=True или resize_image()"
            )

        # Получение метода ресемплинга
        resampling = get_resampling_method(method)

        # === ПРОДВИНУТОЕ УЛУЧШЕНИЕ КАЧЕСТВА ПРИ УВЕЛИЧЕНИИ ===
        # Порядок операций для максимального качества:
        # 1. Noise reduction (до увеличения)
        # 2. Step-up scaling (пошаговое увеличение)
        # 3. Multi-pass sharpening
        # 4. Detail enhancement
        # 5. Color enhancement
        
        result_img = img
        
        # 1. Noise reduction ДО увеличения (на маленьком изображении эффективнее)
        if noise_reduction:
            # Медианный фильтр для подавления шума
            result_img = result_img.filter(ImageFilter.MedianFilter(size=3))
            stats.filters_applied.append("Noise Reduction (median)")
        
        # 2. Увеличение изображения
        target_size = (new_width, new_height)
        
        # Вычисляем фактический scale для определения необходимости step-upscale
        actual_scale = new_width / orig_width if orig_width > 0 else scale_factor
        
        if step_upscale and actual_scale > 2:
            # Пошаговое увеличение для лучшего качества
            result_img = _step_upscale(result_img, target_size, resampling)
            stats.filters_applied.append(f"Step-up scaling ({actual_scale:.1f}x)")
        else:
            # Обычное увеличение
            result_img = result_img.resize(target_size, resample=resampling)
        
        stats.result_size = result_img.size
        
        # 3. Sharpening ПОСЛЕ увеличения (компенсация размытия от ресемплинга)
        if sharpen:
            result_img = _apply_smart_sharpen(result_img, factor=sharpen_factor)
            stats.filters_applied.append(f"Smart Sharpening (f={sharpen_factor})")
        
        # 4. Detail enhancement (улучшение детализации)
        if edge_enhance:
            result_img = _apply_detail_enhancement(result_img)
            stats.filters_applied.append("Detail Enhancement")
        
        # 5. Color enhancement (улучшение цветов)
        if enhance_colors:
            # Лёгкое улучшение контраста и насыщенности для компенсации размытия
            result_img = _enhance_colors(result_img, contrast=1.08, saturation=1.05)
            stats.filters_applied.append("Color Enhancement (c=1.08, s=1.05)")

        # Определение выходного файла
        if output_file is None:
            input_path = Path(input_file)
            suffix = f"_upscaled_{new_width}x{new_height}"
            output_file = str(input_path.with_stem(input_path.stem + suffix))

        # Сохранение
        save_image(result_img, output_file, quality=quality)
        stats.result_file_size = Path(output_file).stat().st_size

        # Проверка размера файла
        if stats.result_file_size > MAX_OUTPUT_FILE_SIZE:
            raise ValueError(
                f"Размер файла ({stats.result_file_size / 1024 / 1024:.1f} MB) "
                f"превышает максимум ({MAX_OUTPUT_FILE_SIZE / 1024 / 1024:.0f} MB)"
            )

        stats.end()

        if return_stats:
            return output_file, stats
        return output_file

    except Exception as e:
        stats.add_error(str(e))
        stats.end()
        raise


def upscale_batch(
    input_files: List[str],
    output_dir: Optional[str] = None,
    scale: float = 2.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    method: str = "lanczos",
    sharpen: bool = True,
    quality: int = 95,
) -> Dict[str, Any]:
    """
    Пакетное увеличение изображений.

    Args:
        input_files: Список путей к файлам.
        output_dir: Директория для сохранения.
        scale: Коэффициент увеличения.
        width: Целевая ширина.
        height: Целевая высота.
        method: Алгоритм ресемплинга.
        sharpen: Применить sharpening.
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

            output_path, stats = upscale_image(
                file_path,
                output_file=output_file,
                scale=scale,
                width=width,
                height=height,
                method=method,
                sharpen=sharpen,
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
