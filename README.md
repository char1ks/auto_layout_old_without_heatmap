Install python3.12.  

Install dependencies:  
```sh
make install
```  

Copy dinov3 weights:  
```sh
cp ~/dinov3-weights/dinov3/* ~/.cache/torch/hub/checkpoints/
```  

## Run  

TODO

```sh
```  

### Eval  

TODO

```sh
!MPLBACKEND=Agg LOG_LEVEL=INFO poetry run python -m cli config.yaml
```  
