"""
Тесты для модуля обработки изображений.
"""

import pytest
from pathlib import Path
from PIL import Image

from parser.image import (
    resize_image,
    upscale_image,
    enhance_image,
    get_image_info,
    SUPPORTED_FORMATS,
)
from parser.image.utils import (
    ImageStats,
    load_image,
    save_image,
    calculate_dimensions,
    get_resampling_method,
    is_supported_image,
)


@pytest.fixture
def test_image(tmp_path):
    """Создание тестового изображения."""
    img_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(str(img_path), "JPEG")
    return str(img_path)


@pytest.fixture
def test_image_png(tmp_path):
    """Создание тестового PNG изображения."""
    img_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (200, 150), color="blue")
    img.save(str(img_path), "PNG")
    return str(img_path)


class TestImageUtils:
    """Тесты для утилит изображений."""

    def test_is_supported_image_valid(self, test_image):
        """Проверка поддержки валидного изображения."""
        assert is_supported_image(test_image) is True

    def test_is_supported_image_invalid(self, tmp_path):
        """Проверка неподдерживаемого формата."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test")
        assert is_supported_image(str(txt_file)) is False

    def test_get_image_info(self, test_image):
        """Проверка получения информации об изображении."""
        info = get_image_info(test_image)
        assert info["width"] == 100
        assert info["height"] == 100
        assert info["format"] == "JPEG"

    def test_get_image_info_not_found(self):
        """Проверка обработки несуществующего файла."""
        with pytest.raises(FileNotFoundError):
            get_image_info("nonexistent.jpg")

    def test_load_image(self, test_image):
        """Проверка загрузки изображения."""
        img = load_image(test_image)
        assert img.size == (100, 100)
        assert img.mode == "RGB"

    def test_save_image(self, tmp_path):
        """Проверка сохранения изображения."""
        img = Image.new("RGB", (50, 50), color="green")
        output_path = str(tmp_path / "output.jpg")
        save_image(img, output_path)
        assert Path(output_path).exists()

    def test_calculate_dimensions_scale(self, test_image):
        """Проверка расчета размеров по scale."""
        img = load_image(test_image)
        w, h = calculate_dimensions(img, scale=0.5)
        assert w == 50
        assert h == 50

    def test_calculate_dimensions_width(self, test_image):
        """Проверка расчета размеров по ширине."""
        img = load_image(test_image)
        w, h = calculate_dimensions(img, width=50)
        assert w == 50
        assert h == 50  # Сохранение пропорций

    def test_calculate_dimensions_height(self, test_image):
        """Проверка расчета размеров по высоте."""
        img = load_image(test_image)
        w, h = calculate_dimensions(img, height=50)
        assert h == 50
        assert w == 50  # Сохранение пропорций

    def test_calculate_dimensions_no_params(self, test_image):
        """Проверка ошибки при отсутствии параметров."""
        img = load_image(test_image)
        with pytest.raises(ValueError):
            calculate_dimensions(img)

    def test_get_resampling_method(self):
        """Проверка получения метода ресемплинга."""
        from PIL import Image
        assert get_resampling_method("lanczos") == Image.LANCZOS
        assert get_resampling_method("bilinear") == Image.BILINEAR
        assert get_resampling_method("bicubic") == Image.BICUBIC
        assert get_resampling_method("nearest") == Image.NEAREST

    def test_image_stats(self, test_image):
        """Проверка статистики обработки."""
        stats = ImageStats()
        stats.start(test_image)
        stats.original_size = (100, 100)
        stats.result_size = (200, 200)
        stats.method = "Resize"
        stats.end()

        assert stats.processing_time >= 0
        assert stats.original_size == (100, 100)


class TestResizeImage:
    """Тесты для изменения размера изображений."""

    def test_resize_by_width(self, test_image, tmp_path):
        """Проверка изменения размера по ширине."""
        output = str(tmp_path / "resized.jpg")
        result_path = resize_image(test_image, output_file=output, width=50)
        
        result_img = Image.open(result_path)
        assert result_img.width == 50
        assert result_img.height == 50

    def test_resize_by_height(self, test_image, tmp_path):
        """Проверка изменения размера по высоте."""
        output = str(tmp_path / "resized.jpg")
        result_path = resize_image(test_image, output_file=output, height=50)
        
        result_img = Image.open(result_path)
        assert result_img.height == 50
        assert result_img.width == 50

    def test_resize_by_scale(self, test_image, tmp_path):
        """Проверка изменения размера по scale."""
        output = str(tmp_path / "resized.jpg")
        result_path = resize_image(test_image, output_file=output, scale=0.5)
        
        result_img = Image.open(result_path)
        assert result_img.width == 50
        assert result_img.height == 50

    def test_resize_upscale(self, test_image, tmp_path):
        """Проверка увеличения размера."""
        output = str(tmp_path / "resized.jpg")
        result_path = resize_image(test_image, output_file=output, scale=2.0)
        
        result_img = Image.open(result_path)
        assert result_img.width == 200
        assert result_img.height == 200

    def test_resize_with_stats(self, test_image, tmp_path):
        """Проверка возврата статистики."""
        output = str(tmp_path / "resized.jpg")
        result_path, stats = resize_image(
            test_image, output_file=output, width=50, return_stats=True
        )
        
        assert stats.original_size == (100, 100)
        assert stats.result_size == (50, 50)
        assert stats.processing_time >= 0

    def test_resize_method_lanczos(self, test_image, tmp_path):
        """Проверка метода Lanczos."""
        output = str(tmp_path / "resized.jpg")
        result_path = resize_image(
            test_image, output_file=output, width=50, method="lanczos"
        )
        assert Path(result_path).exists()

    def test_resize_method_bilinear(self, test_image, tmp_path):
        """Проверка метода Bilinear."""
        output = str(tmp_path / "resized.jpg")
        result_path = resize_image(
            test_image, output_file=output, width=50, method="bilinear"
        )
        assert Path(result_path).exists()


class TestUpscaleImage:
    """Тесты для увеличения изображений."""

    def test_upscale_scale(self, test_image, tmp_path):
        """Проверка увеличения по scale."""
        output = str(tmp_path / "upscaled.jpg")
        result_path = upscale_image(test_image, output_file=output, scale=2.0)
        
        result_img = Image.open(result_path)
        assert result_img.width == 200
        assert result_img.height == 200

    def test_upscale_with_sharpen(self, test_image, tmp_path):
        """Проверка увеличения с sharpening."""
        output = str(tmp_path / "upscaled.jpg")
        result_path = upscale_image(
            test_image, output_file=output, scale=2.0, sharpen=True
        )
        
        result_img = Image.open(result_path)
        assert result_img.width == 200

    def test_upscale_without_sharpen(self, test_image, tmp_path):
        """Проверка увеличения без sharpening."""
        output = str(tmp_path / "upscaled.jpg")
        result_path = upscale_image(
            test_image, output_file=output, scale=2.0, sharpen=False
        )
        
        result_img = Image.open(result_path)
        assert result_img.width == 200

    def test_upscale_by_width(self, test_image, tmp_path):
        """Проверка увеличения по ширине."""
        output = str(tmp_path / "upscaled.jpg")
        result_path = upscale_image(test_image, output_file=output, width=400)
        
        result_img = Image.open(result_path)
        assert result_img.width == 400


class TestEnhanceImage:
    """Тесты для улучшения качества изображений."""

    def test_enhance_default(self, test_image, tmp_path):
        """Проверка улучшения по умолчанию."""
        output = str(tmp_path / "enhanced.jpg")
        result_path = enhance_image(test_image, output_file=output)
        
        assert Path(result_path).exists()
        result_img = Image.open(result_path)
        assert result_img.size == (100, 100)

    def test_enhance_with_sharpen(self, test_image, tmp_path):
        """Проверка улучшения с sharpening."""
        output = str(tmp_path / "enhanced.jpg")
        result_path, stats = enhance_image(
            test_image, output_file=output, sharpen=True, return_stats=True
        )
        
        assert "Sharpen" in str(stats.filters_applied)

    def test_enhance_with_contrast(self, test_image, tmp_path):
        """Проверка улучшения с contrast."""
        output = str(tmp_path / "enhanced.jpg")
        result_path, stats = enhance_image(
            test_image, output_file=output, contrast=True, return_stats=True
        )
        
        assert "Contrast" in str(stats.filters_applied)

    def test_enhance_with_auto_brightness(self, test_image, tmp_path):
        """Проверка улучшения с авто-яркостью."""
        output = str(tmp_path / "enhanced.jpg")
        result_path, stats = enhance_image(
            test_image, output_file=output, auto_brightness=True, return_stats=True
        )
        
        assert "Auto Brightness" in str(stats.filters_applied)

    def test_enhance_with_noise_reduction(self, test_image, tmp_path):
        """Проверка улучшения с шумоподавлением."""
        output = str(tmp_path / "enhanced.jpg")
        result_path = enhance_image(
            test_image, output_file=output, noise_reduction=True
        )
        
        assert Path(result_path).exists()

    def test_quick_enhance(self, test_image, tmp_path):
        """Проверка быстрого улучшения."""
        from parser.image.enhance import quick_enhance
        
        output = str(tmp_path / "enhanced.jpg")
        result_path = quick_enhance(test_image, output_file=output)
        
        assert Path(result_path).exists()


class TestSupportedFormats:
    """Тесты для поддерживаемых форматов."""

    def test_supported_formats_list(self):
        """Проверка списка поддерживаемых форматов."""
        assert ".jpg" in SUPPORTED_FORMATS
        assert ".jpeg" in SUPPORTED_FORMATS
        assert ".png" in SUPPORTED_FORMATS
        assert ".webp" in SUPPORTED_FORMATS
        assert ".bmp" in SUPPORTED_FORMATS
        assert ".tiff" in SUPPORTED_FORMATS

    def test_create_and_test_formats(self, tmp_path):
        """Проверка создания и обработки разных форматов."""
        formats_to_test = [
            (".jpg", "JPEG"),
            (".png", "PNG"),
            (".bmp", "BMP"),
        ]
        
        for ext, fmt in formats_to_test:
            img = Image.new("RGB", (50, 50), color="red")
            img_path = str(tmp_path / f"test{ext}")
            img.save(img_path, fmt)
            
            info = get_image_info(img_path)
            assert info["format"] == fmt
            assert info["width"] == 50
            assert info["height"] == 50
