# YOLOGuiENGINE

![CI](https://github.com/Zenka737/YOLOGuiENGINE/actions/workflows/ci.yml/badge.svg)

Графический интерфейс для работы с YOLO — детекция, обучение, аннотация и управление моделями.

## Возможности

- **Детекция** — запуск в реальном времени с камеры или видеофайла, настройка confidence/IoU
- **Обучение** — запуск тренировки YOLO с выбором гиперпараметров прямо из интерфейса
- **Аннотация** — рисование bounding box'ов мышью, сохранение в формате YOLO `.txt`
- **Менеджер моделей** — скачивание YOLOv8 моделей, импорт своих, удаление

## Установка

```bash
git clone https://github.com/Zenka737/YOLOGuiENGINE.git
cd YOLOGuiENGINE
pip install -r requirements.txt
```

### GPU (NVIDIA CUDA)

`pip install -r requirements.txt` по умолчанию ставит **CPU-версию** PyTorch,
из-за этого программа пишет «GPU не найден», даже если видеокарта есть
(например, RTX 4060). Чтобы включить поддержку CUDA, поставьте GPU-версию
PyTorch **до** установки `requirements.txt`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Проверьте, что CUDA определилась:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Также потребуются актуальные драйверы NVIDIA. На видеокартах серии RTX
(тензорные ядра) детекция и обучение автоматически используют TF32/FP16
для ускорения, если выбрано CUDA-устройство.

### macOS (Apple Silicon)

Программа полностью поддерживает macOS, включая Apple Silicon (M1/M2/M3/M4).
`pip install -r requirements.txt` ставит версию PyTorch со встроенной
поддержкой **MPS** (Metal Performance Shaders) — отдельно ничего доустанавливать
не нужно. При выборе устройства `mps` в настройках обучения используется GPU
Apple.

Проверьте, что MPS определился:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Доступ к камере на macOS запрашивается системой автоматически при первом
запуске детекции — разрешите его в **System Settings → Privacy & Security →
Camera**.

## Запуск

```bash
cd src
python main.py
```

## Требования

- Python 3.9+
- PyQt6
- ultralytics
- opencv-python

## Структура

```
src/
  main.py              # точка входа
  core/
    detector.py        # DetectorThread — инференс в отдельном потоке
  ui/
    main_window.py     # главное окно
    tab_detection.py   # вкладка детекции
    tab_training.py    # вкладка обучения
    tab_annotation.py  # вкладка аннотации
    tab_models.py      # менеджер моделей
    style.qss          # тёмная тема
```

## Разработчик

**@Zenka737**
