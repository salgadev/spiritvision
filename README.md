# mezcal

## sparsifycation
### recipe

**conservative sparse**

```
sparseml.image_classification.train \
    --model_name_or_path models v3.pkl\
    --recipe-path zoo:cv/classification/resnet_v1-18/pytorch/sparseml/imagenet/pruned-conservative/original \
    --pretrained True \
    --arch-key resnet18 \
    --dataset imagenette \
    --dataset-path /PATH/TO/IMAGENETTE  \
    --train-batch-size 128 --test-batch-size 256 \
    --loader-num-workers 8 \
```




