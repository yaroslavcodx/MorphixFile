"""
Модуль супер-разрешения изображений на основе Real-ESRGAN.

Использует предобученные модели Real-ESRGAN для увеличения изображений
с сохранением деталей и улучшением качества.

Поддерживаемые модели:
- RealESRGAN_x4plus - универсальная модель для фотографий (4x увеличение)
- RealESRNet_x4plus - модель с меньшим уровнем артефактов (4x)
- RealESRGAN_x4plus_anime_6B - оптимизирована для аниме и иллюстраций (4x)
- RealESRGAN_x2plus - модель для 2x увеличения
- realesr-animevideov3 - модель для аниме видео (4x, компактная)
- realesr-general-x4v3 - универсальная компактная модель (4x)

Ограничения:
- Максимальный размер выходного изображения: 16384x16384 пикселей
- Требуется GPU (CUDA) для быстрой работы, но поддерживается и CPU
- Минимальный размер входного изображения: 16x16 пикселей
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
import numpy as np

from parser.image.utils import ImageStats, SUPPORTED_FORMATS

# Глобальные константы
MAX_DIMENSION = 16384  # Максимальный размер по одной стороне
MIN_DIMENSION = 16     # Минимальный размер входного изображения

# Модели Real-ESRGAN и их параметры
REAL_ESRGAN_MODELS = {
    "RealESRGAN_x4plus": {
        "scale": 4,
        "description": "Универсальная модель для фотографий (4x)",
        "size": "large",
    },
    "RealESRNet_x4plus": {
        "scale": 4,
        "description": "Модель с меньшим уровнем артефактов (4x)",
        "size": "large",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "scale": 4,
        "description": "Оптимизирована для аниме и иллюстраций (4x)",
        "size": "small",
    },
    "RealESRGAN_x2plus": {
        "scale": 2,
        "description": "Модель для 2x увеличения",
        "size": "large",
    },
    "realesr-animevideov3": {
        "scale": 4,
        "description": "Модель для аниме видео (4x, компактная)",
        "size": "xs",
    },
    "realesr-general-x4v3": {
        "scale": 4,
        "description": "Универсальная компактная модель (4x)",
        "size": "small",
    },
}


def _import_realesrgan():
    """
    Импорт RealESRGANer с проверкой доступности.

    Returns:
        Класс RealESRGANer.

    Raises:
        ImportError: Если Real-ESRGAN не установлен.
    """
    try:
        from realesrgan import RealESRGANer
        return RealESRGANer
    except ImportError as e:
        raise ImportError(
            "Real-ESRGAN не установлен. Установите зависимости:\n"
            "  pip install basicsr>=1.4.2 realesrgan\n"
            "  pip install torch>=1.7 torchvision>=0.8.0\n"
            "  pip install facexlib>=0.2.5 gfpgan>=1.3.5 (опционально, для улучшения лиц)"
        ) from e


def _import_model_class(model_name: str):
    """
    Импорт класса модели для Real-ESRGAN.

    Args:
        model_name: Название модели.

    Returns:
        Кортеж (model_class, netscale, file_urls).
    """
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    if model_name == "RealESRGAN_x4plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    elif model_name == "RealESRNet_x4plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    elif model_name == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    elif model_name == "RealESRGAN_x2plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    elif model_name == "realesr-animevideov3":
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
    elif model_name == "realesr-general-x4v3":
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
        ]
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    return model, netscale, file_url


def super_resolution(
    input_file: str,
    output_file: Optional[str] = None,
    model_name: str = "RealESRGAN_x4plus",
    scale: Optional[float] = None,
    face_enhance: bool = False,
    denoise_strength: float = 0.5,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
    fp32: bool = False,
    gpu_id: Optional[int] = None,
    alpha_upsampler: str = "realesrgan",
    quality: int = 95,
    return_stats: bool = False,
) -> Union[str, Tuple[str, ImageStats]]:
    """
    Увеличение разрешения изображения с использованием Real-ESRGAN.

    Real-ESRGAN использует глубокое обучение для восстановления деталей изображения
    при увеличении, что дает значительно лучшее качество по сравнению с традиционными
    методами интерполяции.

    Args:
        input_file: Путь к входному файлу.
        output_file: Путь к выходному файлу (автогенерация если None).
        model_name: Название модели (RealESRGAN_x4plus, RealESRGAN_x4plus_anime_6B, и т.д.).
        scale: Финальный коэффициент масштабирования (переопределяет масштаб модели).
        face_enhance: Использовать GFPGAN для улучшения лиц.
        denoise_strength: Сила шумоподавления (0-1). Только для realesr-general-x4v3.
        tile: Размер тайла для обработки больших изображений (0 = без тайлинга).
        tile_pad: Размер отступа между тайлами.
        pre_pad: Предварительный отступ для обработки краев.
        fp32: Использовать полную точность (вместо fp16).
        gpu_id: ID GPU для использования (None = автовыбор).
        alpha_upsampler: Метод апсемплинга альфа-канала (realesrgan | bicubic).
        quality: Качество сохранения (1-100) для JPEG/WebP.
        return_stats: Вернуть статистику вместе с путем.

    Returns:
        Путь к сохраненному файлу или кортеж (путь, статистика).

    Raises:
        ImportError: Если Real-ESRGAN не установлен.
        FileNotFoundError: Если файл не найден.
        ValueError: Если параметры некорректны.

    Примеры:
        >>> from parser.image import super_resolution
        >>> # Увеличение фотографии с 4x
        >>> result_path = super_resolution("photo.jpg", model_name="RealESRGAN_x4plus")
        >>>
        >>> # Увеличение аниме изображения
        >>> result_path = super_resolution("anime.png", model_name="RealESRGAN_x4plus_anime_6B")
        >>>
        >>> # Увеличение с улучшением лиц
        >>> result_path = super_resolution("portrait.jpg", face_enhance=True)
        >>>
        >>> # Увеличение с тайлингом для больших изображений
        >>> result_path = super_resolution("large.jpg", tile=512, tile_pad=20)
    """
    from PIL import Image
    import torch

    stats = ImageStats()
    stats.start(input_file)
    stats.method = f"Real-ESRGAN ({model_name})"

    try:
        # Проверка входного файла
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_file}")

        ext = input_path.suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Неподдерживаемый формат: {ext}")

        # Загрузка изображения для проверки размеров
        with Image.open(input_path) as img:
            original_size = img.size
            stats.original_size = original_size

        # Проверка минимального размера
        if original_size[0] < MIN_DIMENSION or original_size[1] < MIN_DIMENSION:
            raise ValueError(
                f"Изображение слишком маленькое: {original_size[0]}x{original_size[1]}. "
                f"Минимум: {MIN_DIMENSION}x{MIN_DIMENSION}"
            )

        # Импорт RealESRGANer
        RealESRGANer = _import_realesrgan()

        # Получение информации о модели
        if model_name not in REAL_ESRGAN_MODELS:
            available = ", ".join(REAL_ESRGAN_MODELS.keys())
            raise ValueError(
                f"Неизвестная модель: {model_name}. Доступные: {available}"
            )

        model_info = REAL_ESRGAN_MODELS[model_name]
        model_scale = model_info["scale"]

        # Определение финального масштаба
        final_scale = scale if scale is not None else model_scale

        # Импорт модели
        model, netscale, file_url = _import_model_class(model_name)

        # Определение пути к модели
        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        model_path = weights_dir / f"{model_name}.pth"

        # Автоматическая загрузка модели если не существует
        if not model_path.exists():
            from basicsr.utils.download_util import load_file_from_url
            urls = file_url if isinstance(file_url, list) else [file_url]
            for url in urls:
                model_path = load_file_from_url(
                    url=url,
                    model_dir=str(weights_dir),
                    progress=True,
                    file_name=f"{model_name}.pth"
                )

        # Определение устройства
        if gpu_id is not None:
            device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Использование half precision если доступно
        use_half = not fp32 and device.type == "cuda"

        # Инициализация RealESRGANer
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=str(model_path),
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=use_half,
            device=device,
            gpu_id=gpu_id,
        )

        # Обработка изображения
        img_cv = np.array(Image.open(input_path).convert("RGB"))
        img_cv = img_cv[:, :, ::-1].copy()  # RGB -> BGR для OpenCV

        # Увеличение с Real-ESRGAN
        output_img, _ = upsampler.enhance(
            img_cv,
            outscale=final_scale,
        )

        # Конвертация обратно в RGB и PIL
        output_img = output_img[:, :, ::-1]  # BGR -> RGB
        result_img = Image.fromarray(output_img)

        # Проверка максимального размера
        if result_img.width > MAX_DIMENSION or result_img.height > MAX_DIMENSION:
            ratio = min(MAX_DIMENSION / result_img.width, MAX_DIMENSION / result_img.height)
            new_size = (int(result_img.width * ratio), int(result_img.height * ratio))
            result_img = result_img.resize(new_size, Image.LANCZOS)
            stats.filters_applied.append(f"Clamped to max: {new_size[0]}x{new_size[1]}")

        stats.result_size = result_img.size

        # Определение выходного файла
        if output_file is None:
            suffix = f"_realesrgan_{model_name}_{result_img.width}x{result_img.height}"
            output_file = str(input_path.with_stem(input_path.stem + suffix))

        # Сохранение результата
        from parser.image.utils import save_image
        save_image(result_img, output_file, quality=quality)
        stats.result_file_size = Path(output_file).stat().st_size

        # Добавление информации о фильтрах
        stats.filters_applied.append(f"Model: {model_name}")
        stats.filters_applied.append(f"Scale: {final_scale}x")
        if face_enhance:
            stats.filters_applied.append("Face Enhancement (GFPGAN)")
        if tile > 0:
            stats.filters_applied.append(f"Tiling: {tile}px")

        stats.end()

        if return_stats:
            return output_file, stats
        return output_file

    except Exception as e:
        stats.add_error(str(e))
        stats.end()
        raise


def super_resolution_batch(
    input_files: List[str],
    output_dir: Optional[str] = None,
    model_name: str = "RealESRGAN_x4plus",
    scale: Optional[float] = None,
    face_enhance: bool = False,
    denoise_strength: float = 0.5,
    tile: int = 0,
    quality: int = 95,
) -> Dict[str, Any]:
    """
    Пакетное увеличение разрешения изображений.

    Args:
        input_files: Список путей к файлам.
        output_dir: Директория для сохранения.
        model_name: Название модели.
        scale: Финальный коэффициент масштабирования.
        face_enhance: Использовать улучшение лиц.
        denoise_strength: Сила шумоподавления.
        tile: Размер тайла для обработки.
        quality: Качество сохранения.

    Returns:
        Словарь с результатами обработки.

    Пример:
        >>> from parser.image import super_resolution_batch
        >>> results = super_resolution_batch(
        ...     ["photo1.jpg", "photo2.jpg"],
        ...     output_dir="enhanced/",
        ...     model_name="RealESRGAN_x4plus"
        ... )
    """
    results = {}

    for file_path in input_files:
        try:
            if output_dir:
                input_path = Path(file_path)
                output_file = str(Path(output_dir) / input_path.name)
            else:
                output_file = None

            output_path, stats = super_resolution(
                file_path,
                output_file=output_file,
                model_name=model_name,
                scale=scale,
                face_enhance=face_enhance,
                denoise_strength=denoise_strength,
                tile=tile,
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


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Получение списка доступных моделей Real-ESRGAN.

    Returns:
        Словарь с информацией о моделях.

    Пример:
        >>> models = get_available_models()
        >>> for name, info in models.items():
        ...     print(f"{name}: {info['description']}")
    """
    return REAL_ESRGAN_MODELS.copy()


def quick_anime_upscale(
    input_file: str,
    output_file: Optional[str] = None,
    scale: Optional[float] = None,
    quality: int = 95,
    return_stats: bool = False,
) -> Union[str, Tuple[str, ImageStats]]:
    """
    Быстрое увеличение аниме изображений с использованием оптимизированной модели.

    Использует модель RealESRGAN_x4plus_anime_6B, которая оптимизирована
    для аниме и иллюстраций с меньшим размером модели.

    Args:
        input_file: Путь к входному файлу.
        output_file: Путь к выходному файлу.
        scale: Финальный коэффициент масштабирования.
        quality: Качество сохранения.
        return_stats: Вернуть статистику.

    Returns:
        Путь к файлу или кортеж (путь, статистика).
    """
    return super_resolution(
        input_file,
        output_file=output_file,
        model_name="RealESRGAN_x4plus_anime_6B",
        scale=scale,
        face_enhance=False,
        quality=quality,
        return_stats=return_stats,
    )


def quick_photo_upscale(
    input_file: str,
    output_file: Optional[str] = None,
    scale: Optional[float] = None,
    face_enhance: bool = False,
    quality: int = 95,
    return_stats: bool = False,
) -> Union[str, Tuple[str, ImageStats]]:
    """
    Быстрое увеличение фотографий с использованием универсальной модели.

    Использует модель RealESRGAN_x4plus для общего улучшения фотографий.
    Опционально может использовать GFPGAN для улучшения лиц.

    Args:
        input_file: Путь к входному файлу.
        output_file: Путь к выходному файлу.
        scale: Финальный коэффициент масштабирования.
        face_enhance: Использовать улучшение лиц.
        quality: Качество сохранения.
        return_stats: Вернуть статистику.

    Returns:
        Путь к файлу или кортет (путь, статистика).
    """
    return super_resolution(
        input_file,
        output_file=output_file,
        model_name="RealESRGAN_x4plus",
        scale=scale,
        face_enhance=face_enhance,
        quality=quality,
        return_stats=return_stats,
    )
