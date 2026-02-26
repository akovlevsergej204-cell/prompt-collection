# 🎨 Image to Prompt (Gradio + BLIP/CLIP)

Веб-приложение на Python + Gradio для генерации качественных AI-промптов по изображению.

## Возможности

- Загрузка изображения через upload, drag & drop и вставку из буфера (`Ctrl+V`)
- 3 режима анализа: fast / classic / detailed
- 10 стилевых пресетов
- Негативный промпт для моделей вроде Stable Diffusion
- Пользовательский суффикс
- Пакетная обработка нескольких изображений

## Стек

- Python 3.10+
- Gradio
- clip-interrogator (BLIP + CLIP)
- torch / torchvision
- Pillow
- transformers
- open-clip-torch

## Структура

```text
image-to-prompt/
├── app.py
├── requirements.txt
├── prompts/
│   └── style_presets.json
└── README.md
```

## Быстрый старт

```bash
cd image-to-prompt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Приложение будет доступно по адресу: <http://localhost:7860>

> При первом запуске `clip-interrogator` загрузит модели (может занять время и потребовать много места в кеше).

## Деплой в Hugging Face Spaces

1. Создай Space с SDK **Gradio**.
2. Загрузи `app.py`, `requirements.txt`, `prompts/style_presets.json`.
3. Дождись авто-сборки и получи публичный URL.
