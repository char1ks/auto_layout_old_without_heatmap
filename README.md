# ML Segmentation Pipeline

## Set up

Install deps:  
```sh
poetry config virtualenvs.in-project true --local
poetry env use 3.11
poetry install

# Установка зависимостей выполняется автоматически через pip install -r requirements.txt
```  

Make examples directories:  
```sh
mkdir -p models input output examples/positive examples/negative
```  

## 🎯 ЗАПУСК АНАЛИЗА

### 🔥 Режим SearchDet:
```bash
python -m searchdet_pipeline.cli.detect \
  --image input/test_metal.jpg \
  --positive examples/positive \
  --negative examples/negative \
  --output output/
```

### ⚡ Только SearchDet (быстрее, меньше памяти):
```bash
python -m searchdet_pipeline.cli.detect \
  --image input/test_metal.jpg \
  --positive examples/positive \
  --negative examples/negative \
  --output output/ \
  --searchdet-only
```

### 📊 С Ground Truth для метрик:
```bash
python -m searchdet_pipeline.cli.detect \
  --image input/test_metal.jpg \
  --positive examples/positive \
  --negative examples/negative \
  --output output/ \
  --ground-truth ground_truth_mask.png \
  --searchdet-only
```

# 3. Установка PyTorch (GPU)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Установка основных зависимостей
!pip install -r requirements.txt

# 5. Установка SearchDet зависимостей
!pip install -r searchdet-main/requirements.txt

# 8. Создание папок
!mkdir -p models input output examples/positive examples/negative
