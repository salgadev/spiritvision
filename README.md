# mezcal

[![train-resnet18](https://github.com/socd06/mezcal/actions/workflows/cml.yml/badge.svg)](https://github.com/socd06/mezcal/actions/workflows/cml.yml)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/socd06/mezcal/HEAD?urlpath=%2Fvoila%2Frender%2Fapp.ipynb)

Multi-Class Image Classification of evaporated mezcal drops using fastai and PyTorch. Web App deployed
on [Binder](https://mybinder.org/v2/gh/socd06/mezcal/HEAD?urlpath=%2Fvoila%2Frender%2Fapp.ipynb).

<img src="icon.png" alt="Agave plant icon" style="height: 240px; width:300px;"/>

## Background

Mezcal is a spirit originally produced in Mexico. There are many varieties of mezcal depending on the agave plant used
its
production, each with its own flavour and intricacies. For example, *espadín* has an earthy flavor as opposed to
*tobalá*, which has subtle smoky notes.

## The Challenge

Mezcal has a very strict and exhaustive quality control authentication and certification method in which sampled drops
of mezcal are chemically analyzed via mass spectrophotometry. In this project, we attempt to authenticate different
mezcal types using image classification models trained in evaporated mezcal drops seen under a microscope.

## Technologies used

Python, fastai, PyTorch, scikit-learn, Voilà

## Getting Started

1. Get and install [Python](https://www.python.org/downloads/) and an IDE of your choice
   (e.g. [PyCharm](https://www.jetbrains.com/pycharm/download/) or [Spyder](https://www.spyder-ide.org/))
2. Make a virtual environment -- *will depend on your IDE* -- (Optional)
3. Install dependencies using the terminal
    ```
    pip install -r requirements.txt
    ```
4. Try out the Jupyter Notebooks and/or the training script

## Collaborating

- Feel free to fork the repo and submit a [Pull Request](https://github.com/socd06/mezcal/compare).
- If you don't have previous experience, please review
  [How to submit a pull request](https://www.freecodecamp.org/news/how-to-submit-a-pull-request-529efe82eea5/)
  and comment on any [Issues](https://github.com/socd06/mezcal/issues) you would like to collaborate.

- Contact [Carlos](mailto:csalgado@uwo.ca) or [Angel](mailto:Angel.reyes@cimat.mx) if you would like to be invited to
  the Discord chat

## References:

- [Revisión del Agave y el Mezcal](https://www.redalyc.org/journal/776/77645907016/)

## torchserve
Archive model like so
```torch-model-archiver --model-name mezcalvision --version 0.1 --model-file .\model.py --serialized-file .\models\resnet_50_2023May08_0614PM.pt --handler image_classifier --extra-files ./index_to_name.
json
```

Deploy with the following command:
```
 torchserve --start --model-store model_store --models mezcalvision=mezcalvision.mar --ncs
```

