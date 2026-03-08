"""
Модуль обработки изображений.

Поддерживает:
- Изменение размера (resize, downscale)
- Увеличение (upscale)
- Улучшение качества (enhance)
- Супер-разрешение на основе AI (Real-ESRGAN)
- Пакетную обработку
"""

from parser.image.resize import resize_image, downscale_image, resize_batch
from parser.image.upscale import upscale_image, upscale_batch
from parser.image.enhance import enhance_image, enhance_batch, quick_enhance
from parser.image.esrgan import (
    super_resolution,
    super_resolution_batch,
    get_available_models,
    quick_anime_upscale,
    quick_photo_upscale,
    REAL_ESRGAN_MODELS,
)
from parser.image.utils import (
    ImageStats,
    SUPPORTED_FORMATS,
    get_image_info,
    load_image,
    save_image,
    find_images,
)

__version__ = "1.0.0"
__all__ = [
    # Resize
    "resize_image",
    "downscale_image",
    "resize_batch",
    # Upscale
    "upscale_image",
    "upscale_batch",
    # Enhance
    "enhance_image",
    "enhance_batch",
    "quick_enhance",
    # Real-ESRGAN
    "super_resolution",
    "super_resolution_batch",
    "get_available_models",
    "quick_anime_upscale",
    "quick_photo_upscale",
    "REAL_ESRGAN_MODELS",
    # Utils
    "ImageStats",
    "SUPPORTED_FORMATS",
    "get_image_info",
    "load_image",
    "save_image",
    "find_images",
]
