# ML Segmentation Pipeline

## Set up

Install deps:  
```sh
poetry config virtualenvs.in-project true --local
poetry env use 3.11
poetry install

git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA && pip install -e . && cd ..

git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e . && cd ..
```  

Make examples directories:  
```sh
mkdir -p models input output examples/positive examples/negative
```  

## 🎯 ЗАПУСК АНАЛИЗА

### 🔥 Гибридный режим (LLaVA + SearchDet):
```bash
PYTHONPATH=. poetry run python hybrid_searchdet_pipeline.py \
  --image input/test_metal.jpg \
  --positive examples/positive \
  --negative examples/negative \
  --output output/
```

### ⚡ Только SearchDet (быстрее, меньше памяти):
```bash
PYTHONPATH=. poetry run python hybrid_searchdet_pipeline.py \
  --image input/test_metal.jpg \
  --positive examples/positive \
  --negative examples/negative \
  --output output/ \
  --searchdet-only
```

### 📊 С Ground Truth для метрик:
```bash
PYTHONPATH=. poetry run python hybrid_searchdet_pipeline.py \
  --image input/test_metal.jpg \
  --positive examples/positive \
  --negative examples/negative \
  --output output/ \
  --ground-truth ground_truth_mask.png \
  --searchdet-only
```
