Install python3.12.  

Install dependencies:  
```sh
pip install poetry
poetry install
```  

## Tests

Требуется GPU (CUDA).А также скокировать веса:
Copy dinov3 weights:  
```sh
cp ~/dinov3-weights/dinov3/* ~/.cache/torch/hub/checkpoints/
```  
Запуск локально тестов:
```sh
make tests
```

## Run  

TODO

```sh
```  

### Eval  

TODO

```sh
!poetry run python -m 'evaluation.serchdet_detector_v1.main'
```
