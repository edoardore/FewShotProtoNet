# FewShotProtoNet
Progetto di Machine Learning su Few-Shot learning con il metodo di meta learning Prototypical Network.

## Setting
Impostare nei successivi moduli .py su quale GPU si vuole lavorare
```python
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"     #0 o 1 se si dispone di due GPU
```

## train.py
Eseguire il modulo train.py per addestrare e condurre un nuovo esperimento, le successive variabili possono essere modificate per eseguire training in scenari differenti
```python
    # .Conv4, .Conv6, .ResNet10, .ResNet18, .ResNet34
    model = 'Conv6'

    # CUB, miniImagenet, cross, omniglot, cross_char
    dataset = 'CUB'

    # class num to classify for training
    train_n_way = 5

    # class num to classify for testing (validation)
    test_n_way = 5

    # number of labeled data in each class, same as n_support
    n_shot = 1

    # perform data augmentation or not during training
    train_aug = True

    # Save frequency
    save_freq = 50

    # Starting epoch
    start_epoch = 0

    # Stopping epoch
    stop_epoch = -1

    # continue from previous trained model with largest epoch
    resume = False
```
A termine dell'esecuzione si avrà nella directory checkpoints il miglior modello best_model.tar (non presenti nella repo a causa delle grandi dimensioni).

## save_features.py
Regolando le variabili in modo analogo a quanto visto prima in fase di training è necessario estrarre le features dal modello in .tar in un file .hdf5 presente a fine esecuzione nella directory features

## test.py
Dopo aver addestrato i modelli per CUB nelle configurazioni (1S , 5S)x(NoAugmentation, Augmentation)x(Conv4, Conv6, ResNet10, ResNet18, ResNet34).
Quelli per Omniglot con 1Shot, 5Shot sempre con Conv4, Omniglot->EMNIST con 1Shot, 5Shot sempre con Conv4.
Infine CUB ResNet18 5W-5S NoAdaptation e CUB ResNet18 5W-5S Adaptation, risulta possibile eseguire il file test.py che mostra in output le seguenti tabelle e grafici riportati in relazione.


### Esempio di output del file ```test.py```
```
ssh://er******@m***.d****.u***.**:*****/u*/b*/python3 -u /home/er******/test.py
Test omniglot, Conv4, n_shot = 1
600 Test Acc = 97.92% +- 0.28%
Test omniglot, Conv4, n_shot = 5
600 Test Acc = 99.37% +- 0.12%
Test cross_char, Conv4, n_shot = 1
600 Test Acc = 73.29% +- 0.78%
Test cross_char, Conv4, n_shot = 5
600 Test Acc = 86.95% +- 0.58%
--------------------------------------------------------------------
Omni 5W-1S     Omni 5W-5S     Omni->Emnist 5W-1S  Omni->Emnist 5W-5S
--------------------------------------------------------------------
97.92 +- 0.28  99.37 +- 0.12    73.29 +- 0.78       86.95 +- 0.58
--------------------------------------------------------------------
Test without augmentiation, Conv4, n_shot = 1
600 Test Acc = 52.16% +- 0.92%
Test without augmentiation, Conv6, n_shot = 1
600 Test Acc = 53.20% +- 0.95%
Test without augmentiation, ResNet10, n_shot = 1
600 Test Acc = 59.97% +- 0.93%
Test without augmentiation, ResNet18, n_shot = 1
600 Test Acc = 61.52% +- 0.97%
Test without augmentiation, ResNet34, n_shot = 1
600 Test Acc = 59.69% +- 0.98%
Test without augmentiation, Conv4, n_shot = 5
600 Test Acc = 67.39% +- 0.72%
Test without augmentiation, Conv6, n_shot = 5
600 Test Acc = 67.71% +- 0.74%
Test without augmentiation, ResNet10, n_shot = 5
600 Test Acc = 72.46% +- 0.73%
Test without augmentiation, ResNet18, n_shot = 5
600 Test Acc = 73.66% +- 0.69%
Test without augmentiation, ResNet34, n_shot = 5
600 Test Acc = 74.43% +- 0.72%
Test with augmentiation, Conv4, n_shot = 1
600 Test Acc = 50.86% +- 0.92%
Test with augmentiation, Conv6, n_shot = 1
600 Test Acc = 65.67% +- 1.01%
Test with augmentiation, ResNet10, n_shot = 1
600 Test Acc = 73.16% +- 0.87%
Test with augmentiation, ResNet18, n_shot = 1
600 Test Acc = 74.18% +- 0.90%
Test with augmentiation, ResNet34, n_shot = 1
600 Test Acc = 74.56% +- 0.92%
Test with augmentiation, Conv4, n_shot = 5
600 Test Acc = 76.37% +- 0.69%
Test with augmentiation, Conv6, n_shot = 5
600 Test Acc = 81.74% +- 0.61%
Test with augmentiation, ResNet10, n_shot = 5
600 Test Acc = 85.83% +- 0.49%
Test with augmentiation, ResNet18, n_shot = 5
600 Test Acc = 86.51% +- 0.51%
Test with augmentiation, ResNet34, n_shot = 5
600 Test Acc = 87.91% +- 0.46%
------------------------------------------------------------------------------------------
Backbone:        Conv4          Conv6          ResNet10       ResNet18       ResNet34     
------------------------------------------------------------------------------------------
CUB 5W-1S NoAug  52.16 +- 0.92  53.2 +- 0.95   59.97 +- 0.93  61.52 +- 0.97  59.69 +- 0.98
CUB 5W-5S NoAug  67.39 +- 0.72  67.71 +- 0.74  72.46 +- 0.73  73.66 +- 0.69  74.43 +- 0.72
CUB 5W-1S Aug    50.86 +- 0.92  65.67 +- 1.01  73.16 +- 0.87  74.18 +- 0.9   74.56 +- 0.92
CUB 5W-5S Aug    76.37 +- 0.69  81.74 +- 0.61  85.83 +- 0.49  86.77 +- 0.49   87.91 +- 0.46
------------------------------------------------------------------------------------------
Test CUB, ResNet18, n_shot = 5, adaptation = False
600 Test Acc = 86.77% +- 0.49%
Test CUB, ResNet18, n_shot = 5, adaptation = True
600 Test Acc = 86.51% +- 0.50%
--------------------------------------------------------------
CUB ResNet18 5W-5S NoAdaptation  CUB ResNet18 5W-5S Adaptation
--------------------------------------------------------------
     86.77 +- 0.49                     86.51 +- 0.5 
--------------------------------------------------------------

Process finished with exit code 0

```

## Librerie utilizzate
```python
import numpy as np
import torch
import torch.optim
import glob
import os
import torch.utils.data.sampler
import random
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional
from PIL import ImageEnhance
import torchvision.transforms as transforms
import json
import h5py
```

## Riferimenti
Tutti i datasets di questo progetto sono stati presi da [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz), [Omniglot](https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip?raw=true),[Omniglot2](https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip?raw=true), [EMNIST](https://github.com/NanqingD/DAOSL/raw/master/data/emnist.zip). A supporto delle immagini sono stati realizzati tre file json (base, val, novel) con i campi:
```json
{"label_names": ["class0","class1","..."], "image_names": ["filepath1","filepath2","..."],"image_labels":["l1","l2","l3","..."]}  
```
L'implementazione è stata ripresa e riadattata dalle seguenti repo: [Gestione data e backbone](https://github.com/facebookresearch/low-shot-shrink-hallucinate), [ProtoNet e gestione Omniglot](https://github.com/jakesnell/prototypical-networks), [Struttura e setup](https://github.com/wyharveychen/CloserLookFewShot)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.


## License
[Edoardo Re](https://github.com/edoardore), 2021
