"""
CLI интерфейс для универсального парсера файлов.
"""

import sys
import json
import glob as glob_module
from pathlib import Path
from typing import Optional, List

import click

from parser import FileParser, detect_format, detect_encoding
from parser.utils import format_size, ParseStats
from parser.image import resize_image, upscale_image, enhance_image, get_image_info, SUPPORTED_FORMATS
from parser.image.esrgan import super_resolution, super_resolution_batch, get_available_models
from parser.image.utils import find_images


@click.group()
@click.version_option(version="1.0.0", prog_name="parser")
def main():
    """
    Universal File Parser - Универсальный парсер файлов.

    Поддерживает парсинг, конвертацию и анализ различных типов файлов.
    """
    pass


@main.command()
@click.option("--file", "-f", "file_path", help="Путь к файлу для парсинга")
@click.option("--files", "-F", "files_pattern", help="Шаблон для нескольких файлов (например, *.json)")
@click.option("--output", "-o", "output_path", help="Путь для сохранения результата")
@click.option("--output-dir", "-d", "output_dir", help="Директория для сохранения результатов")
@click.option("--format", "format_name", help="Формат файла (автоопределение если не указан)")
@click.option("--encoding", "-e", default=None, help="Кодировка файла")
@click.option("--filter-keys", "-k", help="Ключи для фильтрации (через запятую)")
@click.option("--regex", "-r", help="Регулярное выражение для фильтрации")
@click.option("--start-line", type=int, help="Начальная строка")
@click.option("--end-line", type=int, help="Конечная строка")
@click.option("--quiet", "-q", is_flag=True, help="Тихий режим (без вывода статистики)")
@click.option("--pretty", "-p", is_flag=True, help="Красивый вывод JSON")
def parse(
    file_path: Optional[str],
    files_pattern: Optional[str],
    output_path: Optional[str],
    output_dir: Optional[str],
    format_name: Optional[str],
    encoding: Optional[str],
    filter_keys: Optional[str],
    regex: Optional[str],
    start_line: Optional[int],
    end_line: Optional[int],
    quiet: bool,
    pretty: bool,
):
    """
    Парсинг файла.

    Примеры:

        parser parse --file data.csv --output data.json

        parser parse --files "*.json" --output-dir parsed/

        parser parse --file config.yaml --filter-keys "database,server"
    """
    parser = FileParser()

    # Определение списка файлов
    files = _get_files(file_path, files_pattern)

    if not files:
        click.echo("Ошибка: Не указаны файлы или файлы не найдены", err=True)
        sys.exit(1)

    # Обработка ключей фильтрации
    filter_keys_list = None

    if filter_keys:
        filter_keys_list = [k.strip() for k in filter_keys.split(",")]

    results = {}

    for fp in files:
        try:
            data = parser.parse(
                fp,
                encoding=encoding,
                format_name=format_name,
                filter_keys=filter_keys_list,
                regex=regex,
                start_line=start_line,
                end_line=end_line,
            )

            results[fp] = data

            # Сохранение
            if output_path and len(files) == 1:
                _save_result(data, output_path, pretty)
            elif output_dir:
                out_file = Path(output_dir) / f"{Path(fp).stem}.json"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                _save_result(data, str(out_file), pretty)

            # Вывод в консоль
            if not output_path and not output_dir:
                if len(files) > 1:
                    click.echo(f"\n=== {fp} ===")

                if isinstance(data, (dict, list)):
                    click.echo(json.dumps(data, indent=2 if pretty else None, ensure_ascii=False))
                else:
                    click.echo(str(data)[:1000])  # Ограничение вывода

            # Статистика
            if not quiet and parser.last_stats:
                _print_stats(parser.last_stats, verbose=len(files) == 1)

        except Exception as e:
            click.echo(f"Ошибка при парсинге {fp}: {str(e)}", err=True)
            results[fp] = {"error": str(e)}

    # Сохранение всех результатов
    if output_path and len(files) > 1:
        _save_result(results, output_path, pretty)


@main.command()
@click.option("--file", "-f", "input_file", required=True, help="Входной файл")
@click.option("--from", "from_format", help="Входной формат")
@click.option("--to", "to_format", required=True, help="Выходной формат")
@click.option("--output", "-o", "output_file", help="Выходной файл")
@click.option("--encoding", "-e", default=None, help="Кодировка")
def convert(
    input_file: str,
    from_format: Optional[str],
    to_format: str,
    output_file: Optional[str],
    encoding: Optional[str],
):
    """
    Конвертация между форматами.

    Примеры:

        parser convert --file config.json --to yaml --output config.yaml

        parser convert --file data.csv --to xlsx --output data.xlsx

        parser convert --file page.html --to md --output page.md
    """
    parser = FileParser()

    try:
        result_file = parser.convert(
            input_file,
            to_format,
            output_file,
            encoding=encoding,
        )

        click.echo(f"Файл сконвертирован: {result_file}")

        if parser.last_stats:
            _print_stats(parser.last_stats)

    except Exception as e:
        click.echo(f"Ошибка конвертации: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option("--file", "-f", "file_path", required=True, help="Файл для анализа")
@click.option("--filter", "filter_pattern", help="Паттерн для фильтрации строк")
@click.option("--regex", "-r", help="Регулярное выражение")
@click.option("--start-line", type=int, help="Начальная строка")
@click.option("--end-line", type=int, help="Конечная строка")
@click.option("--stats", "-s", is_flag=True, help="Только статистика")
def analyze(
    file_path: str,
    filter_pattern: Optional[str],
    regex: Optional[str],
    start_line: Optional[int],
    end_line: Optional[int],
    stats: bool,
):
    """
    Анализ файла.

    Примеры:

        parser analyze --file log.txt --filter "ERROR"

        parser analyze --file data.csv --stats

        parser analyze --file app.log --regex "Exception.*"
    """
    parser = FileParser()

    try:
        result = parser.analyze(
            file_path,
            filter_pattern=filter_pattern,
            regex=regex,
            start_line=start_line,
            end_line=end_line,
        )

        if stats:
            click.echo("=== Статистика ===")
            click.echo(f"Файл: {result['file']}")
            click.echo(f"Формат: {result['format']}")
            click.echo(f"Кодировка: {result['encoding']}")
            click.echo(f"Размер: {format_size(result['size'])}")
            click.echo(f"Записей: {result['records']}")
            click.echo(f"Время парсинга: {result['parse_time']:.3f}s")
        else:
            click.echo("=== Результаты анализа ===")
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        click.echo(f"Ошибка анализа: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option("--file", "-f", "file_path", required=True, help="Файл для проверки")
@click.option("--encoding", "-e", is_flag=True, help="Проверить кодировку")
@click.option("--structure", "-s", is_flag=True, help="Показать структуру")
def validate(
    file_path: str,
    encoding: bool,
    structure: bool,
):
    """
    Валидация файла.

    Примеры:

        parser validate --file data.json

        parser validate --file config.yaml --encoding
    """
    parser = FileParser()

    detected_format = detect_format(file_path)
    detected_encoding = detect_encoding(file_path)

    click.echo("=== Информация о файле ===")
    click.echo(f"Путь: {file_path}")
    click.echo(f"Размер: {format_size(Path(file_path).stat().st_size)}")
    click.echo(f"Формат: {detected_format or 'не определен'}")
    click.echo(f"Кодировка: {detected_encoding}")

    if structure:
        try:
            data = parser.parse(file_path)

            if isinstance(data, dict):
                click.echo(f"\nКлючи верхнего уровня: {list(data.keys())}")
            elif isinstance(data, list):
                click.echo(f"\nКоличество записей: {len(data)}")

                if data:
                    click.echo(f"Структура первой записи: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")

        except Exception as e:
            click.echo(f"Ошибка при чтении структуры: {str(e)}", err=True)


@main.command()
def formats():
    """
    Список поддерживаемых форматов.
    """
    parser = FileParser()
    supported = parser.get_supported_formats()

    click.echo("=== Поддерживаемые форматы ===\n")

    for format_name, extensions in sorted(supported.items()):
        click.echo(f"{format_name}: {', '.join(extensions)}")


@main.command()
@click.argument("archive_path")
@click.option("--output", "-o", "output_dir", help="Директория для распаковки")
@click.option("--list", "-l", "list_only", is_flag=True, help="Только список файлов")
def extract(archive_path: str, output_dir: Optional[str], list_only: bool):
    """
    Распаковка архива.

    Примеры:

        parser extract archive.zip

        parser extract archive.zip --output extracted/

        parser extract archive.tar.gz --list
    """
    parser = FileParser()

    try:
        format_name = detect_format(archive_path)

        if format_name not in ("zip", "tar", "gzip"):
            click.echo(f"Ошибка: {archive_path} не является поддерживаемым архивом", err=True)
            sys.exit(1)

        data = parser.parse(archive_path, list_only=list_only, extract_path=output_dir)

        if list_only or not output_dir:
            if isinstance(data, dict):
                click.echo(f"Всего файлов: {data.get('total_files', 0)}")

                for file_info in data.get("files", []):
                    size = file_info.get("size", 0)
                    name = file_info.get("filename") or file_info.get("name")
                    click.echo(f"  {name} ({format_size(size) if size else '0 B'})")
            elif isinstance(data, list):
                for item in data:
                    click.echo(f"  {item}")

        if output_dir:
            click.echo(f"Архив распакован в: {output_dir}")

    except Exception as e:
        click.echo(f"Ошибка: {str(e)}", err=True)
        sys.exit(1)


def _get_files(
    file_path: Optional[str],
    files_pattern: Optional[str],
) -> List[str]:
    """Получение списка файлов."""
    files = []

    if file_path:
        files.append(file_path)

    if files_pattern:
        files.extend(glob_module.glob(files_pattern))

    return files


def _save_result(data, output_path: str, pretty: bool) -> None:
    """Сохранение результата."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, (dict, list)):
        indent = 2 if pretty else None
        content = json.dumps(data, indent=indent, ensure_ascii=False)
    else:
        content = str(data)

    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def _print_stats(stats: ParseStats, verbose: bool = True) -> None:
    """Вывод статистики."""
    if verbose:
        click.echo(f"\n=== Статистика ===")
        click.echo(f"Время парсинга: {stats.parse_time:.3f}s")
        click.echo(f"Записей: {stats.records_count}")
        click.echo(f"Ошибок: {stats.errors_count}")

        if stats.errors:
            click.echo("Ошибки:")
            for error in stats.errors:
                click.echo(f"  - {error}")
    else:
        click.echo(f"  [{stats.records_count} записей, {stats.parse_time:.3f}s]")


# === Команды для обработки изображений ===

@main.group()
def image():
    """
    Обработка изображений.

    Поддерживает resize, upscale, enhance.
    """
    pass


@image.command()
@click.option("--file", "-f", "file_path", help="Путь к изображению")
@click.option("--folder", "-F", "folder_path", help="Папка с изображениями для пакетной обработки")
@click.option("--width", "-w", type=int, help="Целевая ширина в пикселях")
@click.option("--height", "-h", type=int, help="Целевая высота в пикселях")
@click.option("--scale", "-s", type=float, help="Коэффициент масштабирования (0.5 = 50%, 2 = 200%)")
@click.option("--method", "-m", type=click.Choice(["nearest", "bilinear", "bicubic", "lanczos"]), default="lanczos", help="Алгоритм ресемплинга")
@click.option("--output", "-o", "output_path", help="Путь для сохранения результата")
@click.option("--output-dir", "-d", "output_dir", help="Директория для сохранения результатов")
@click.option("--quality", "-q", type=int, default=95, help="Качество сохранения (1-100)")
@click.option("--quiet", is_flag=True, help="Тихий режим (без вывода статистики)")
def resize(
    file_path: Optional[str],
    folder_path: Optional[str],
    width: Optional[int],
    height: Optional[int],
    scale: Optional[float],
    method: str,
    output_path: Optional[str],
    output_dir: Optional[str],
    quality: int,
    quiet: bool,
):
    """
    Изменение размера изображения.

    Примеры:

        parser image resize --file photo.jpg --width 800

        parser image resize --file photo.jpg --scale 0.5

        parser image resize --folder ./images --width 1024 --output-dir ./resized/
    """
    if not file_path and not folder_path:
        click.echo("Ошибка: Укажите --file или --folder", err=True)
        sys.exit(1)

    # Получение списка файлов
    files = []
    if file_path:
        files.append(file_path)
    if folder_path:
        files.extend(find_images(folder_path, recursive=False))

    if not files:
        click.echo("Ошибка: Файлы не найдены", err=True)
        sys.exit(1)

    for fp in files:
        try:
            if len(files) > 1 and output_dir:
                out_file = str(Path(output_dir) / Path(fp).name)
            elif len(files) == 1 and output_path:
                out_file = output_path
            else:
                out_file = None

            result_path, stats = resize_image(
                fp,
                output_file=out_file,
                width=width,
                height=height,
                scale=scale,
                method=method,
                quality=quality,
                return_stats=True,
            )

            if not quiet:
                click.echo(f"Обработано: {Path(fp).name}")
                click.echo(f"  Исходный размер: {stats.original_size[0]}x{stats.original_size[1]}")
                click.echo(f"  Новый размер: {stats.result_size[0]}x{stats.result_size[1]}")
                click.echo(f"  Метод: {stats.method}")
                click.echo(f"  Время: {stats.processing_time:.3f}s")

        except Exception as e:
            click.echo(f"Ошибка при обработке {fp}: {str(e)}", err=True)


@image.command()
@click.option("--file", "-f", "file_path", required=True, help="Путь к изображению")
@click.option("--scale", "-s", type=float, default=2.0, help="Коэффициент увеличения (1.01-16, 2 = 200%)")
@click.option("--width", "-w", type=int, help="Целевая ширина (переопределяет scale), макс. 16384")
@click.option("--height", "-h", type=int, help="Целевая высота (переопределяет scale), макс. 16384")
@click.option("--method", "-m", type=click.Choice(["nearest", "bilinear", "bicubic", "lanczos"]), default="lanczos", help="Алгоритм ресемплинга")
@click.option("--sharpen", is_flag=True, default=True, help="Применить sharpening после увеличения")
@click.option("--sharpen-factor", type=float, default=2.0, help="Коэффициент sharpening (1.0-3.0)")
@click.option("--noise-reduction", is_flag=True, default=False, help="Применить шумоподавление")
@click.option("--edge-enhance", is_flag=True, default=False, help="Применить улучшение краёв")
@click.option("--enhance-colors", is_flag=True, default=True, help="Улучшить контраст и насыщенность")
@click.option("--step-upscale", is_flag=True, default=True, help="Пошаговое увеличение (лучше качество)")
@click.option("--output", "-o", "output_path", help="Путь для сохранения результата")
@click.option("--quality", "-q", type=int, default=95, help="Качество сохранения (1-100)")
@click.option("--quiet", is_flag=True, help="Тихий режим (без вывода статистики)")
def upscale(
    file_path: str,
    scale: float,
    width: Optional[int],
    height: Optional[int],
    method: str,
    sharpen: bool,
    sharpen_factor: float,
    noise_reduction: bool,
    edge_enhance: bool,
    enhance_colors: bool,
    step_upscale: bool,
    output_path: Optional[str],
    quality: int,
    quiet: bool,
):
    """
    Увеличение изображения (upscale) с улучшением качества.

    Примеры:

        parser image upscale --file photo.jpg --scale 2

        parser image upscale --file photo.jpg --width 3840 --height 2160

        parser image upscale --file photo.jpg --scale 4 --sharpen --edge-enhance
        
        parser image upscale --file photo.jpg --scale 4 --step-upscale --enhance-colors
    """
    try:
        result_path, stats = upscale_image(
            file_path,
            output_file=output_path,
            scale=scale,
            width=width,
            height=height,
            method=method,
            sharpen=sharpen,
            sharpen_factor=sharpen_factor,
            noise_reduction=noise_reduction,
            edge_enhance=edge_enhance,
            enhance_colors=enhance_colors,
            step_upscale=step_upscale,
            quality=quality,
            return_stats=True,
        )

        if not quiet:
            click.echo(f"Обработано: {Path(file_path).name}")
            click.echo(f"  Исходный размер: {stats.original_size[0]}x{stats.original_size[1]}")
            click.echo(f"  Новый размер: {stats.result_size[0]}x{stats.result_size[1]}")
            click.echo(f"  Метод: {stats.method}")
            if stats.filters_applied:
                click.echo(f"  Фильтры: {', '.join(stats.filters_applied)}")
            click.echo(f"  Время: {stats.processing_time:.3f}s")

    except Exception as e:
        click.echo(f"Ошибка при обработке {file_path}: {str(e)}", err=True)
        sys.exit(1)


@image.command()
@click.option("--file", "-f", "file_path", required=True, help="Путь к изображению")
@click.option("--sharpen", is_flag=True, default=True, help="Применить sharpening")
@click.option("--sharpen-factor", type=float, default=1.5, help="Коэффициент sharpening")
@click.option("--noise-reduction", is_flag=True, help="Применить шумоподавление")
@click.option("--contrast", is_flag=True, default=True, help="Применить улучшение контраста")
@click.option("--contrast-factor", type=float, default=1.1, help="Коэффициент контраста")
@click.option("--auto-brightness", is_flag=True, default=True, help="Авто-коррекция яркости")
@click.option("--color", is_flag=True, help="Применить улучшение цвета")
@click.option("--output", "-o", "output_path", help="Путь для сохранения результата")
@click.option("--quality", "-q", type=int, default=95, help="Качество сохранения (1-100)")
@click.option("--quiet", is_flag=True, help="Тихий режим (без вывода статистики)")
def enhance(
    file_path: str,
    sharpen: bool,
    sharpen_factor: float,
    noise_reduction: bool,
    contrast: bool,
    contrast_factor: float,
    auto_brightness: bool,
    color: bool,
    output_path: Optional[str],
    quality: int,
    quiet: bool,
):
    """
    Улучшение качества изображения.

    Примеры:

        parser image enhance --file photo.jpg

        parser image enhance --file photo.jpg --sharpen --contrast --auto-brightness
    """
    try:
        result_path, stats = enhance_image(
            file_path,
            output_file=output_path,
            sharpen=sharpen,
            sharpen_factor=sharpen_factor,
            noise_reduction=noise_reduction,
            contrast=contrast,
            contrast_factor=contrast_factor,
            auto_brightness=auto_brightness,
            color=color,
            quality=quality,
            return_stats=True,
        )

        if not quiet:
            click.echo(f"Обработано: {Path(file_path).name}")
            click.echo(f"  Размер: {stats.result_size[0]}x{stats.result_size[1]}")
            click.echo(f"  Фильтры: {', '.join(stats.filters_applied) if stats.filters_applied else 'Нет'}")
            click.echo(f"  Время: {stats.processing_time:.3f}s")

    except Exception as e:
        click.echo(f"Ошибка при обработке {file_path}: {str(e)}", err=True)
        sys.exit(1)


@image.command()
@click.option("--file", "-f", "file_path", required=True, help="Путь к изображению")
def info(file_path: str):
    """
    Получение информации об изображении.

    Примеры:

        parser image info --file photo.jpg
    """
    try:
        info = get_image_info(file_path)

        click.echo("=== Информация об изображении ===")
        click.echo(f"Файл: {info['file']}")
        click.echo(f"Формат: {info['format']}")
        click.echo(f"Режим: {info['mode']}")
        click.echo(f"Размер: {info['width']}x{info['height']}")
        click.echo(f"Размер файла: {info['file_size_formatted']}")

    except Exception as e:
        click.echo(f"Ошибка: {str(e)}", err=True)
        sys.exit(1)


@image.command()
def formats():
    """
    Список поддерживаемых форматов изображений.
    """
    click.echo("=== Поддерживаемые форматы изображений ===")
    for fmt in SUPPORTED_FORMATS:
        click.echo(f"  {fmt}")


@image.command()
@click.option("--file", "-f", "file_path", help="Путь к изображению")
@click.option("--folder", "-F", "folder_path", help="Папка с изображениями для пакетной обработки")
@click.option("--model", "-m", "model_name", type=click.Choice(list(get_available_models().keys())), default="RealESRGAN_x4plus", help="Модель Real-ESRGAN")
@click.option("--scale", "-s", type=float, help="Финальный коэффициент масштабирования (переопределяет масштаб модели)")
@click.option("--face-enhance", is_flag=True, default=False, help="Использовать GFPGAN для улучшения лиц")
@click.option("--denoise-strength", type=float, default=0.5, help="Сила шумоподавления (0-1). Только для realesr-general-x4v3")
@click.option("--tile", type=int, default=0, help="Размер тайла для обработки больших изображений (0 = без тайлинга)")
@click.option("--tile-pad", type=int, default=10, help="Размер отступа между тайлами")
@click.option("--pre-pad", type=int, default=0, help="Предварительный отступ для обработки краев")
@click.option("--fp32", is_flag=True, default=False, help="Использовать полную точность (вместо fp16)")
@click.option("--gpu-id", type=int, default=None, help="ID GPU для использования (None = автовыбор)")
@click.option("--output", "-o", "output_path", help="Путь для сохранения результата")
@click.option("--output-dir", "-d", "output_dir", help="Директория для сохранения результатов")
@click.option("--quality", "-q", type=int, default=95, help="Качество сохранения (1-100)")
@click.option("--quiet", is_flag=True, help="Тихий режим (без вывода статистики)")
def super_resolution_cmd(
    file_path: Optional[str],
    folder_path: Optional[str],
    model_name: str,
    scale: Optional[float],
    face_enhance: bool,
    denoise_strength: float,
    tile: int,
    tile_pad: int,
    pre_pad: int,
    fp32: bool,
    gpu_id: Optional[int],
    output_path: Optional[str],
    output_dir: Optional[str],
    quality: int,
    quiet: bool,
):
    """
    Увеличение разрешения изображения с использованием Real-ESRGAN (AI).

    Real-ESRGAN использует глубокое обучение для восстановления деталей изображения
    при увеличении, что дает значительно лучшее качество по сравнению с традиционными
    методами интерполяции.

    Примеры:

        parser image super-resolution --file photo.jpg --model RealESRGAN_x4plus

        parser image super-resolution --file anime.png --model RealESRGAN_x4plus_anime_6B

        parser image super-resolution --file portrait.jpg --face-enhance

        parser image super-resolution --folder ./photos --output-dir ./enhanced/
    """
    if not file_path and not folder_path:
        click.echo("Ошибка: Укажите --file или --folder", err=True)
        sys.exit(1)

    # Получение списка файлов
    files = []
    if file_path:
        files.append(file_path)
    if folder_path:
        files.extend(find_images(folder_path, recursive=False))

    if not files:
        click.echo("Ошибка: Файлы не найдены", err=True)
        sys.exit(1)

    # Проверка доступности модели
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "CPU"
        if not quiet:
            click.echo(f"Используется устройство: {device}")
            if device == "CPU":
                click.echo("Предупреждение: Обработка на CPU может быть медленной. Рекомендуется GPU с CUDA.")
    except ImportError:
        pass

    for fp in files:
        try:
            # Определение выходного файла
            if len(files) > 1 and output_dir:
                input_path = Path(fp)
                out_file = str(Path(output_dir) / input_path.name)
            elif len(files) == 1 and output_path:
                out_file = output_path
            else:
                out_file = None

            result_path, stats = super_resolution(
                fp,
                output_file=out_file,
                model_name=model_name,
                scale=scale,
                face_enhance=face_enhance,
                denoise_strength=denoise_strength,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                fp32=fp32,
                gpu_id=gpu_id,
                quality=quality,
                return_stats=True,
            )

            if not quiet:
                click.echo(f"\nОбработано: {Path(fp).name}")
                click.echo(f"  Исходный размер: {stats.original_size[0]}x{stats.original_size[1]}")
                click.echo(f"  Новый размер: {stats.result_size[0]}x{stats.result_size[1]}")
                click.echo(f"  Модель: {model_name}")
                if stats.filters_applied:
                    click.echo(f"  Параметры: {', '.join(stats.filters_applied)}")
                click.echo(f"  Время: {stats.processing_time:.3f}s")

        except Exception as e:
            click.echo(f"Ошибка при обработке {fp}: {str(e)}", err=True)
            if not quiet:
                import traceback
                traceback.print_exc()


@image.command()
@click.option("--list-models", is_flag=True, help="Список доступных моделей")
def models(list_models: bool):
    """
    Информация о моделях Real-ESRGAN.
    
    Примеры:
    
        parser image models --list-models
    """
    if list_models:
        click.echo("=== Модели Real-ESRGAN ===\n")
        models = get_available_models()
        for name, info in models.items():
            click.echo(f"{name}:")
            click.echo(f"  Описание: {info['description']}")
            click.echo(f"  Масштаб: {info['scale']}x")
            click.echo(f"  Размер: {info['size']}")
            click.echo()
    else:
        click.echo("Используйте --list-models для просмотра доступных моделей")


if __name__ == "__main__":
    main()
