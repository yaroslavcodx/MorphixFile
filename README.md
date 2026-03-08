# Universal File Parser

Универсальный инструмент для парсинга, конвертации и анализа различных типов файлов.

## Статус

![Tests](https://img.shields.io/badge/tests-147%20passed-green)
![Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

## Возможности

### Поддерживаемые форматы (17+)

| Категория | Форматы |
|-----------|---------|
| Текстовые | `.txt`, `.md`, `.csv`, `.tsv`, `.log` |
| Табличные | `.xlsx`, `.ods` |
| Данные | `.json`, `.geojson`, `.yaml`, `.yml`, `.toml`, `.ini` |
| Веб | `.xml`, `.html` |
| Бинарные | `.bin` (hex-view) |
| Архивы | `.zip`, `.tar`, `.gz`, `.tgz` |
| **Изображения** | **`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`** |

### Функции

- **Парсинг** — чтение и вывод структуры файлов в удобном формате
- **Конвертация** — CSV ↔ Excel, JSON ↔ YAML, HTML ↔ Markdown
- **Фильтрация** — по ключам, полям, регулярным выражениям, диапазону строк
- **Пакетная обработка** — работа с несколькими файлами одновременно
- **Автоопределение формата** — по расширению и содержимому
- **Поддержка кодировок** — UTF-8, UTF-16, Windows-1251
- **Логирование** — статистика операций
- **Обработка изображений** — изменение размера, увеличение, улучшение качества

## Установка

```bash
# Установка из исходников
pip install -e .

# Или через pip
pip install universal-file-parser

# С поддержкой обработки изображений (Real-ESRGAN)
pip install -e ".[image]"
```

### Зависимости

**Основные:**
- `openpyxl>=3.1.0` — работа с Excel
- `odfpy>=1.4.1` — работа с ODS
- `pyyaml>=6.0` — парсинг YAML
- `toml>=0.10.2` — парсинг TOML
- `click>=8.1.0` — CLI фреймворк
- `chardet>=5.0.0` — определение кодировки
- `Pillow>=10.0.0` — обработка изображений
- `numpy>=1.24.0` — работа с массивами
- `opencv-python>=4.5.0` — компьютерное зрение

**Опциональные (для AI супер-разрешения):**
- `torch>=2.1.0` — PyTorch
- `torchvision>=0.16.0` — компьютерное зрение
- `basicsr>=1.4.2` — базовая библиотека для супер-разрешения
- `realesrgan>=0.3.0` — Real-ESRGAN
- `facexlib>=0.2.5` — улучшение лиц
- `gfpgan>=1.3.5` — восстановление лиц

## Использование

### CLI

#### Парсинг файла

```bash
# Парсинг CSV в JSON
parser parse --file data.csv --output data.json

# Парсинг с выводом в консоль
parser parse --file config.yaml

# Пакетная обработка
parser parse --files *.json --output-dir parsed/
```

#### Конвертация

```bash
# JSON в YAML
parser convert --from json --to yaml --file config.json --output config.yaml

# CSV в Excel
parser convert --from csv --to xlsx --file data.csv --output data.xlsx

# HTML в Markdown
parser convert --from html --to md --file page.html --output page.md
```

#### Анализ

```bash
# Анализ лог-файла с фильтрацией
parser analyze --file log.txt --filter "ERROR"

# Анализ с диапазоном строк
parser analyze --file data.csv --start 10 --end 100

# Анализ с регулярным выражением
parser analyze --file app.log --regex "Exception.*"
```

#### Обработка изображений

```bash
# Изменение размера
parser image resize --file photo.jpg --width 800
parser image resize --file photo.jpg --scale 50

# Увеличение с улучшением качества
parser image upscale --file photo.jpg --scale 2

# Улучшение качества (sharpening, contrast, brightness)
parser image enhance --file photo.jpg

# Информация об изображении
parser image info --file photo.jpg

# Пакетная обработка
parser image resize --folder ./images --width 1024 --output-dir ./resized/
```

#### AI супер-разрешение (Real-ESRGAN)

```bash
# Увеличение фотографии с 4x
parser image super-resolution --file photo.jpg --model RealESRGAN_x4plus

# Увеличение аниме изображения
parser image super-resolution --file anime.png --model RealESRGAN_x4plus_anime_6B

# Увеличение с улучшением лиц
parser image super-resolution --file portrait.jpg --face-enhance

# Пакетная обработка
parser image super-resolution --folder ./photos --output-dir ./enhanced/

# Список доступных моделей
parser image models --list-models
```

### Модели Real-ESRGAN

| Модель | Масштаб | Описание | Рекомендация |
|--------|---------|----------|-------------|
| `RealESRGAN_x4plus` | 4x | Универсальная для фотографий | ✅ Лучшая для фото |
| `RealESRNet_x4plus` | 4x | Меньше артефактов, мягче | ⚠️ Может быть размыто |
| `RealESRGAN_x4plus_anime_6B` | 4x | Для аниме и иллюстраций | ❌ Не для фото |
| `RealESRGAN_x2plus` | 2x | Для 2x увеличения | ✅ Для меньшего увеличения |
| `realesr-animevideov3` | 4x | Для аниме видео | ❌ Не для фото |
| `realesr-general-x4v3` | 4x | Универсальная с шумоподавлением | ⚠️ Можно с --denoise-strength |

### Python API

```python
from parser import FileParser
from parser.formats import CsvParser, JsonParser

# Использование абстрактного парсера
parser = FileParser()

# Автоматическое определение формата
data = parser.parse("data.csv")
print(data)

# Конвертация
parser.convert("config.json", "yaml", "config.yaml")

# Использование конкретных парсеров
csv_parser = CsvParser()
data = csv_parser.parse("data.csv")

json_parser = JsonParser()
json_data = json_parser.parse("data.json")
```

#### Фильтрация данных

```python
from parser import FileParser

parser = FileParser()

# Фильтрация по ключу
data = parser.parse("data.json", filter_keys=["name", "age"])

# Фильтрация по регулярному выражению
data = parser.parse("log.txt", regex=r"ERROR.*")

# Фильтрация по диапазону строк
data = parser.parse("data.csv", start_line=10, end_line=100)
```

#### Пакетная обработка

```python
from parser import FileParser

parser = FileParser()

# Обработка нескольких файлов
results = parser.parse_batch(["file1.json", "file2.json", "file3.json"])

# Сохранение результатов
for filename, data in results.items():
    parser.save(data, f"output/{filename}")
```

#### Обработка изображений

```python
from parser.image import resize_image, upscale_image, enhance_image, get_image_info

# Изменение размера
resize_image("photo.jpg", width=800)
resize_image("photo.jpg", scale=0.5)  # 50% от оригинала

# Увеличение с улучшением качества
upscale_image("photo.jpg", scale=2.0, sharpen=True)

# Улучшение качества
enhance_image(
    "photo.jpg",
    sharpen=True,
    contrast=True,
    auto_brightness=True,
    noise_reduction=False,
)

# Информация об изображении
info = get_image_info("photo.jpg")
print(f"Размер: {info['width']}x{info['height']}")
```

#### AI супер-разрешение (Real-ESRGAN)

```python
from parser.image import super_resolution, get_available_models

# Увеличение фотографии с 4x
result_path = super_resolution("photo.jpg", model_name="RealESRGAN_x4plus")

# Увеличение аниме изображения
result_path = super_resolution("anime.png", model_name="RealESRGAN_x4plus_anime_6B")

# С возвратом статистики
result_path, stats = super_resolution("photo.jpg", return_stats=True)
print(f"Увеличено с {stats.original_size} до {stats.result_size}")

# Пакетная обработка
from parser.image import super_resolution_batch
results = super_resolution_batch(
    ["photo1.jpg", "photo2.jpg"],
    output_dir="enhanced/",
    model_name="RealESRGAN_x4plus"
)

# Список доступных моделей
models = get_available_models()
for name, info in models.items():
    print(f"{name}: {info['description']}")
```

## Структура проекта

```
universal-file-parser/
├── parser/
│   ├── __init__.py
│   ├── cli.py
│   ├── base.py
│   ├── utils.py
│   ├── parser.py
│   ├── image/              # Модуль обработки изображений
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   ├── resize.py
│   │   ├── upscale.py
│   │   ├── enhance.py
│   │   └── esrgan.py       # Real-ESRGAN интеграция
│   └── formats/
│       ├── __init__.py
│       ├── text_parser.py
│       ├── csv_parser.py
│       ├── json_parser.py
│       ├── yaml_parser.py
│       ├── xml_parser.py
│       ├── html_parser.py
│       ├── xlsx_parser.py
│       ├── ods_parser.py
│       ├── toml_parser.py
│       ├── ini_parser.py
│       ├── bin_parser.py
│       └── archive_parser.py
├── tests/
│   ├── __init__.py
│   ├── test_all_formats.py
│   ├── test_parser.py
│   ├── test_csv.py
│   ├── test_json.py
│   ├── test_yaml.py
│   ├── test_xml.py
│   ├── test_utils.py
│   ├── test_image.py         # Тесты обработки изображений
│   ├── test_esrgan.py        # Тесты Real-ESRGAN
│   └── data/                 # Тестовые файлы
├── examples/                  # Примеры использования
├── pyproject.toml             # Конфигурация проекта
├── setup.py
├── requirements.txt
├── pytest.ini
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

## Расширение (плагины)

Для добавления поддержки нового формата создайте класс, наследующий `BaseParser`:

```python
from parser.base import BaseParser

class MyFormatParser(BaseParser):
    extensions = [".myformat"]
    
    def parse(self, file_path: str, encoding: str = "utf-8") -> dict:
        # Реализация парсинга
        pass
    
    def save(self, data: dict, file_path: str) -> None:
        # Реализация сохранения
        pass
```

Зарегистрируйте парсер в `parser/formats/__init__.py`.

## Тесты

```bash
# Запуск всех тестов
pytest

# Запуск с покрытием
pytest --cov=parser

# Запуск конкретного теста
pytest tests/test_json.py
```

## Лицензия

MIT License

## Вклад в проект

1. Fork репозиторий
2. Создайте ветку (`git checkout -b feature/amazing-feature`)
3. Закоммитьте изменения (`git commit -m 'Add amazing feature'`)
4. Отправьте в ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request
