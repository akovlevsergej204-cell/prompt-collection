import json
import os

import gradio as gr
from PIL import Image
from clip_interrogator import Config, Interrogator

# ========== КОНФИГ ==========

STYLE_PRESETS_PATH = os.path.join(os.path.dirname(__file__), "prompts", "style_presets.json")


def load_style_presets():
    with open(STYLE_PRESETS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


STYLE_PRESETS = load_style_presets()

# ========== ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ==========

# clip-interrogator: BLIP для описания + CLIP для стилевых тегов
# Используем ViT-L-14/openai — баланс скорости и качества
# Для GPU: caption_model_name="blip-large"
# Для CPU: caption_model_name="blip-base"

ci = None


def get_interrogator(use_large: bool = False):
    """Ленивая инициализация модели (загрузка при первом вызове)."""
    global ci
    if ci is None:
        config = Config(
            clip_model_name="ViT-L-14/openai",
            caption_model_name="blip-large" if use_large else "blip-base",
            caption_max_length=64,
            chunk_size=2048,
            quiet=False,
        )
        ci = Interrogator(config)
    return ci


# ========== ОСНОВНЫЕ ФУНКЦИИ ==========

def analyze_image(image, mode, style, negative_prompt_enabled, custom_suffix):
    """
    Основная функция анализа изображения.

    Args:
        image: PIL Image — загруженное изображение
        mode: str — режим анализа ("fast", "classic", "detailed")
        style: str — ключ стиля из пресетов
        negative_prompt_enabled: bool — добавлять ли негативный промпт
        custom_suffix: str — пользовательский суффикс

    Returns:
        tuple: (базовый_промпт, улучшенный_промпт, негативный_промпт)
    """
    if image is None:
        return "⚠️ Загрузи изображение", "", ""

    image = image.convert("RGB")
    interrogator = get_interrogator()

    if mode == "fast":
        base_prompt = interrogator.interrogate_fast(image)
    elif mode == "classic":
        base_prompt = interrogator.interrogate_classic(image)
    else:
        base_prompt = interrogator.interrogate(image)

    enhanced_prompt = base_prompt
    if style and style in STYLE_PRESETS:
        enhanced_prompt = base_prompt + STYLE_PRESETS[style]["suffix"]

    if custom_suffix and custom_suffix.strip():
        enhanced_prompt = enhanced_prompt + ", " + custom_suffix.strip()

    negative_prompt = ""
    if negative_prompt_enabled:
        negative_prompt = (
            "ugly, blurry, low quality, distorted, deformed, disfigured, "
            "bad anatomy, bad proportions, extra limbs, mutated hands, "
            "poorly drawn face, watermark, signature, text, logo, "
            "oversaturated, underexposed, cropped, worst quality, "
            "low resolution, jpeg artifacts, duplicate"
        )

    return base_prompt, enhanced_prompt, negative_prompt


def batch_analyze(files, mode, style):
    """
    Пакетная обработка нескольких изображений.

    Args:
        files: list — список файлов
        mode: str — режим анализа
        style: str — ключ стиля

    Returns:
        str: результат в формате CSV-подобного текста
    """
    if not files:
        return "⚠️ Загрузи хотя бы одно изображение"

    results = []
    for index, file in enumerate(files, start=1):
        image = Image.open(file).convert("RGB")
        base, enhanced, _ = analyze_image(image, mode, style, False, "")
        results.append(f"--- Изображение {index} ---\n🔹 Базовый: {base}\n🔸 Улучшенный: {enhanced}\n")

    return "\n".join(results)


# ========== ИНТЕРФЕЙС GRADIO ==========

style_choices = [(value["label"], key) for key, value in STYLE_PRESETS.items()]

with gr.Blocks(
    title="🎨 Image → Prompt Generator",
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="gray",
    ),
    css="""
        .main-header { text-align: center; margin-bottom: 16px; }
        .prompt-box textarea { font-size: 14px !important; line-height: 1.6 !important; }
        footer { display: none !important; }
    """,
) as app:

    gr.HTML(
        """
        <div class="main-header">
            <h1>🎨 Image → Prompt Generator</h1>
            <p style="color: gray;">Загрузи изображение — получи промпт для Midjourney, Stable Diffusion, Flux, DALL-E</p>
        </div>
    """
    )

    with gr.Tabs():
        with gr.TabItem("🖼️ Одно изображение"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="Загрузи изображение",
                        sources=["upload", "clipboard"],
                        height=350,
                    )

                    mode_select = gr.Radio(
                        choices=[
                            ("⚡ Быстрый (только BLIP)", "fast"),
                            ("🎯 Классический (BLIP + CLIP)", "classic"),
                            ("🔬 Детальный (полный анализ)", "detailed"),
                        ],
                        value="classic",
                        label="Режим анализа",
                    )

                    style_select = gr.Dropdown(
                        choices=style_choices,
                        value="photo",
                        label="Стиль промпта",
                    )

                    with gr.Accordion("⚙️ Дополнительные настройки", open=False):
                        negative_enabled = gr.Checkbox(
                            label="Добавить негативный промпт",
                            value=True,
                        )
                        custom_suffix = gr.Textbox(
                            label="Свой суффикс (добавится к промпту)",
                            placeholder="masterpiece, best quality, ultra detailed...",
                            lines=2,
                        )

                    generate_btn = gr.Button(
                        "✨ Сгенерировать промпт",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=1):
                    base_output = gr.Textbox(
                        label="🔹 Базовый промпт",
                        lines=4,
                        show_copy_button=True,
                        elem_classes=["prompt-box"],
                    )
                    enhanced_output = gr.Textbox(
                        label="🔸 Улучшенный промпт (со стилем)",
                        lines=6,
                        show_copy_button=True,
                        elem_classes=["prompt-box"],
                    )
                    negative_output = gr.Textbox(
                        label="🚫 Негативный промпт",
                        lines=3,
                        show_copy_button=True,
                        elem_classes=["prompt-box"],
                    )

            generate_btn.click(
                fn=analyze_image,
                inputs=[image_input, mode_select, style_select, negative_enabled, custom_suffix],
                outputs=[base_output, enhanced_output, negative_output],
            )

        with gr.TabItem("📦 Пакетная обработка"):
            gr.Markdown("Загрузи несколько изображений — получи промпты для всех сразу.")

            batch_files = gr.File(
                label="Загрузи изображения (несколько файлов)",
                file_count="multiple",
                file_types=["image"],
            )

            with gr.Row():
                batch_mode = gr.Radio(
                    choices=[
                        ("⚡ Быстрый", "fast"),
                        ("🎯 Классический", "classic"),
                        ("🔬 Детальный", "detailed"),
                    ],
                    value="fast",
                    label="Режим",
                )
                batch_style = gr.Dropdown(
                    choices=style_choices,
                    value="photo",
                    label="Стиль",
                )

            batch_btn = gr.Button("📦 Обработать все", variant="primary")
            batch_output = gr.Textbox(
                label="Результаты",
                lines=15,
                show_copy_button=True,
            )

            batch_btn.click(
                fn=batch_analyze,
                inputs=[batch_files, batch_mode, batch_style],
                outputs=[batch_output],
            )

        with gr.TabItem("📖 Инструкция"):
            gr.Markdown(
                """
## Как пользоваться

### Режимы анализа
| Режим | Скорость | Качество | Описание |
|-------|----------|----------|----------|
| ⚡ Быстрый | ~2 сек | Базовое | Только BLIP — короткое описание сцены |
| 🎯 Классический | ~10 сек | Хорошее | BLIP + CLIP — описание + стилевые теги |
| 🔬 Детальный | ~30 сек | Максимальное | Полный анализ с подбором художников, медиумов, трендов |

### Стили
Каждый стиль добавляет профессиональные теги к базовому промпту:
- **📸 Фотография** — реалистичный фото-стиль
- **🎨 Digital Art** — цифровая иллюстрация
- **🎬 Cinematic** — кинематографический стиль
- **🖌️ Масляная живопись** — классическая живопись
- **🕯️ Свечной арт** — для скульптурных свечей и восковых форм
- И другие...

### Советы
1. **Для Midjourney** — используй режим «Классический» + стиль по выбору
2. **Для Stable Diffusion** — используй «Детальный» + негативный промпт
3. **Для Flux** — «Быстрый» режим даёт хорошие результаты
4. **Свой суффикс** — добавь свои любимые теги в настройках
5. **Ctrl+V** — можно вставить изображение прямо из буфера обмена
            """
            )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
