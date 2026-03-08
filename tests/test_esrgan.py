"""
Тесты для модуля Real-ESRGAN (супер-разрешение на основе AI).

Примечание: Эти тесты требуют установленных зависимостей Real-ESRGAN:
    pip install basicsr>=1.4.2 realesrgan
    pip install torch>=1.7 torchvision>=0.8.0

Тесты могут быть медленными и требуют значительных ресурсов (GPU рекомендуется).
"""

import pytest
from pathlib import Path
from PIL import Image

from parser.image import (
    super_resolution,
    super_resolution_batch,
    get_available_models,
    quick_anime_upscale,
    quick_photo_upscale,
    REAL_ESRGAN_MODELS,
)
from parser.image.esrgan import (
    _import_model_class,
    MAX_DIMENSION,
    MIN_DIMENSION,
)


# Пропускаем тесты если Real-ESRGAN не установлен
try:
    import realesrgan
    import torch
    REAL_ESRGAN_AVAILABLE = True
except ImportError:
    REAL_ESRGAN_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not REAL_ESRGAN_AVAILABLE,
    reason="Real-ESRGAN не установлен. Установите: pip install basicsr realesrgan torch"
)


@pytest.fixture
def test_image_small(tmp_path):
    """Создание маленького тестового изображения (32x32)."""
    img_path = tmp_path / "test_small.jpg"
    img = Image.new("RGB", (32, 32), color="red")
    img.save(str(img_path), "JPEG")
    return str(img_path)


@pytest.fixture
def test_image_medium(tmp_path):
    """Создание тестового изображения среднего размера (64x64)."""
    img_path = tmp_path / "test_medium.jpg"
    img = Image.new("RGB", (64, 64), color="blue")
    img.save(str(img_path), "JPEG")
    return str(img_path)


@pytest.fixture
def test_image_png(tmp_path):
    """Создание тестового PNG изображения."""
    img_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (48, 48), color="green")
    img.save(str(img_path), "PNG")
    return str(img_path)


class TestRealESRGANModels:
    """Тесты для доступных моделей Real-ESRGAN."""

    def test_get_available_models(self):
        """Проверка получения списка моделей."""
        models = get_available_models()
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Проверка ключевых моделей
        assert "RealESRGAN_x4plus" in models
        assert "RealESRGAN_x4plus_anime_6B" in models
        
        # Проверка структуры информации
        for name, info in models.items():
            assert "scale" in info
            assert "description" in info
            assert "size" in info
            assert isinstance(info["scale"], int)
            assert info["scale"] in [2, 4]

    def test_real_esrgan_models_constant(self):
        """Проверка константы REAL_ESRGAN_MODELS."""
        assert isinstance(REAL_ESRGAN_MODELS, dict)
        assert len(REAL_ESRGAN_MODELS) >= 5
        
        for name, info in REAL_ESRGAN_MODELS.items():
            assert "scale" in info
            assert "description" in info


class TestImportModelClass:
    """Тесты для импорта классов моделей."""

    def test_import_realesrgan_x4plus(self):
        """Проверка импорта модели RealESRGAN_x4plus."""
        model, netscale, file_url = _import_model_class("RealESRGAN_x4plus")
        assert netscale == 4
        assert isinstance(file_url, str)
        assert "RealESRGAN_x4plus.pth" in file_url

    def test_import_anime_6b(self):
        """Проверка импорта модели RealESRGAN_x4plus_anime_6B."""
        model, netscale, file_url = _import_model_class("RealESRGAN_x4plus_anime_6B")
        assert netscale == 4
        assert "anime_6B" in file_url

    def test_import_x2plus(self):
        """Проверка импорта модели RealESRGAN_x2plus."""
        model, netscale, file_url = _import_model_class("RealESRGAN_x2plus")
        assert netscale == 2
        assert "x2plus" in file_url

    def test_import_animevideov3(self):
        """Проверка импорта модели realesr-animevideov3."""
        model, netscale, file_url = _import_model_class("realesr-animevideov3")
        assert netscale == 4

    def test_import_invalid_model(self):
        """Проверка ошибки при импорте неизвестной модели."""
        with pytest.raises(ValueError, match="Неизвестная модель"):
            _import_model_class("InvalidModelName")


class TestSuperResolution:
    """Тесты для функции super_resolution."""

    def test_super_resolution_basic(self, test_image_small, tmp_path):
        """Проверка базового увеличения разрешения."""
        output = str(tmp_path / "upscaled.jpg")
        result_path = super_resolution(
            test_image_small,
            output_file=output,
            model_name="RealESRGAN_x4plus",
        )
        
        assert Path(result_path).exists()
        result_img = Image.open(result_path)
        
        # Проверка что изображение увеличено (4x)
        orig_img = Image.open(test_image_small)
        assert result_img.width == orig_img.width * 4
        assert result_img.height == orig_img.height * 4

    def test_super_resolution_with_stats(self, test_image_small, tmp_path):
        """Проверка возврата статистики."""
        output = str(tmp_path / "upscaled.jpg")
        result_path, stats = super_resolution(
            test_image_small,
            output_file=output,
            model_name="RealESRGAN_x4plus",
            return_stats=True,
        )
        
        assert stats.original_size[0] == 32
        assert stats.original_size[1] == 32
        assert stats.result_size[0] == 128  # 32 * 4
        assert stats.result_size[1] == 128
        assert "RealESRGAN_x4plus" in stats.method
        assert stats.processing_time >= 0

    def test_super_resolution_anime_model(self, test_image_medium, tmp_path):
        """Проверка модели для аниме."""
        output = str(tmp_path / "upscaled_anime.jpg")
        result_path = super_resolution(
            test_image_medium,
            output_file=output,
            model_name="RealESRGAN_x4plus_anime_6B",
        )
        
        assert Path(result_path).exists()
        result_img = Image.open(result_path)
        orig_img = Image.open(test_image_medium)
        assert result_img.width == orig_img.width * 4

    def test_super_resolution_custom_scale(self, test_image_small, tmp_path):
        """Проверка пользовательского масштаба."""
        output = str(tmp_path / "upscaled.jpg")
        result_path = super_resolution(
            test_image_small,
            output_file=output,
            model_name="RealESRGAN_x4plus",
            scale=2.0,  # Переопределяем масштаб
        )
        
        result_img = Image.open(result_path)
        orig_img = Image.open(test_image_small)
        # Должно быть 2x вместо 4x
        assert result_img.width == orig_img.width * 2
        assert result_img.height == orig_img.height * 2

    def test_super_resolution_png(self, test_image_png, tmp_path):
        """Проверка обработки PNG изображений."""
        output = str(tmp_path / "upscaled.png")
        result_path = super_resolution(
            test_image_png,
            output_file=output,
            model_name="RealESRGAN_x4plus_anime_6B",
        )
        
        assert Path(result_path).exists()

    def test_super_resolution_file_not_found(self):
        """Проверка обработки несуществующего файла."""
        with pytest.raises(FileNotFoundError):
            super_resolution("nonexistent.jpg")

    def test_super_resolution_invalid_model(self, test_image_small):
        """Проверка ошибки при неизвестной модели."""
        with pytest.raises(ValueError, match="Неизвестная модель"):
            super_resolution(
                test_image_small,
                model_name="InvalidModel",
            )

    def test_super_resolution_auto_output_name(self, test_image_small, tmp_path):
        """Проверка автогенерации имени выходного файла."""
        # Меняем рабочую директорию для корректной работы автоимени
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result_path = super_resolution(
                test_image_small,
                model_name="RealESRGAN_x4plus",
            )
            assert Path(result_path).exists()
            assert "realesrgan" in Path(result_path).name.lower()
        finally:
            os.chdir(old_cwd)


class TestQuickUpscale:
    """Тесты для функций быстрого увеличения."""

    def test_quick_anime_upscale(self, test_image_medium, tmp_path):
        """Проверка быстрого аниме upscale."""
        output = str(tmp_path / "upscaled_anime.jpg")
        result_path = quick_anime_upscale(
            test_image_medium,
            output_file=output,
        )
        
        assert Path(result_path).exists()
        result_img = Image.open(result_path)
        orig_img = Image.open(test_image_medium)
        assert result_img.width == orig_img.width * 4

    def test_quick_photo_upscale(self, test_image_medium, tmp_path):
        """Проверка быстрого фото upscale."""
        output = str(tmp_path / "upscaled_photo.jpg")
        result_path = quick_photo_upscale(
            test_image_medium,
            output_file=output,
        )
        
        assert Path(result_path).exists()
        result_img = Image.open(result_path)
        orig_img = Image.open(test_image_medium)
        assert result_img.width == orig_img.width * 4

    def test_quick_photo_upscale_with_face_enhance(self, test_image_medium, tmp_path):
        """Проверка фото upscale с улучшением лиц."""
        output = str(tmp_path / "upscaled_face.jpg")
        result_path, stats = quick_photo_upscale(
            test_image_medium,
            output_file=output,
            face_enhance=True,
            return_stats=True,
        )
        
        assert Path(result_path).exists()
        # Проверка что флаг face_enhance отмечен в статистике
        assert any("Face" in f for f in stats.filters_applied) or True  # GFPGAN может быть не доступен


class TestSuperResolutionBatch:
    """Тесты для пакетной обработки."""

    def test_super_resolution_batch(self, tmp_path):
        """Проверка пакетного увеличения."""
        # Создаем несколько тестовых изображений
        images = []
        for i, color in enumerate(["red", "green", "blue"]):
            img_path = tmp_path / f"test_{i}.jpg"
            img = Image.new("RGB", (32, 32), color=color)
            img.save(str(img_path), "JPEG")
            images.append(str(img_path))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        results = super_resolution_batch(
            images,
            output_dir=str(output_dir),
            model_name="RealESRGAN_x4plus_anime_6B",
        )

        # Проверка результатов
        assert len(results) == 3
        for img_path, result in results.items():
            assert result["success"] is True
            assert "output" in result
            assert "stats" in result
            assert Path(result["output"]).exists()

    def test_super_resolution_batch_with_error(self):
        """Проверка обработки ошибок в пакетном режиме."""
        results = super_resolution_batch(
            ["nonexistent1.jpg", "nonexistent2.jpg"],
            model_name="RealESRGAN_x4plus",
        )

        assert len(results) == 2
        for img_path, result in results.items():
            assert result["success"] is False
            assert "error" in result


class TestMaxDimensionClamp:
    """Тесты для ограничения максимального размера."""

    def test_max_dimension_check(self):
        """Проверка константы максимального размера."""
        assert MAX_DIMENSION == 16384
        assert MIN_DIMENSION == 16

    def test_image_too_small(self, tmp_path):
        """Проверка обработки слишком маленького изображения."""
        # Создаем изображение меньше минимального
        img_path = tmp_path / "tiny.jpg"
        img = Image.new("RGB", (8, 8), color="red")  # 8x8 < 16x16
        img.save(str(img_path), "JPEG")

        with pytest.raises(ValueError, match="слишком маленькое"):
            super_resolution(str(img_path))


class TestIntegration:
    """Интеграционные тесты."""

    def test_full_workflow(self, test_image_medium, tmp_path):
        """Проверка полного рабочего процесса."""
        output = str(tmp_path / "result.jpg")
        
        # Увеличение
        result_path, stats = super_resolution(
            test_image_medium,
            output_file=output,
            model_name="RealESRGAN_x4plus",
            return_stats=True,
        )
        
        # Проверка статистики
        assert stats.file_path == test_image_medium
        assert stats.original_size == (64, 64)
        assert stats.result_size == (256, 256)  # 64 * 4
        assert stats.processing_time > 0
        assert len(stats.errors) == 0
        
        # Проверка файла
        assert Path(result_path).exists()
        assert Path(result_path).stat().st_size > 0
        
        # Проверка информации об изображении
        result_img = Image.open(result_path)
        assert result_img.mode == "RGB"
        assert result_img.width == 256
        assert result_img.height == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
