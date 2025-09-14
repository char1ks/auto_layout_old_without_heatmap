# ML Segmentation Pipeline

## Set up

Install deps:  
```sh
poetry config virtualenvs.in-project true --local
poetry env use 3.11
poetry install
```  

Or: 
```sh
pytnon3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```  

Install dinov3:
```sh
УСТАНОВИТЬ DINOV3 (gh pull --> pip install -e .)

```  

```sh
export PYTHONPATH=$PYTHONPATH:${PWD}/vendor/dinov3
```  

Make examples directories:  
```sh
mkdir -p models input output examples/positive examples/negative
```  

Copy dinov3 weights:  
```sh
cp ~/dinov3-weights/dinov3/* ~/.cache/torch/hub/checkpoints/
```  

## Run  

```sh
!python -m searchdet_pipeline.cli.detect detect foto00117.jpg.png \
    --positive examples/positive/ \
    --output output/ \
    --dinov3-backbone vit7b16 \
    --vit-pooling cls 
```  
