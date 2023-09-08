# pickled-mustard-tuber
Automatic detection of cortical fibers on pickled mustard tuber using Multispectral and High-Definition imaging combined with a newly developed deep learning model
## programs
'run.py' is a program that trains or tests model.

'model.py' is a model program.

'dataset.py' is a data loading program.

'Fusion.py' is a program that fuses HD images with MS images using guilded filtering.

'Evaluation.py' is a program that evaluats predicted results with recall, precision, dice, and outputs a related csv file in 'data/Evaluation_result'.
## folder
'model' is trained model file.
'data' is a data folder,includes:
'HD'---- HD images

'ms_843' ---- MS imgaes use bands8, 4, 3 as false-color images

'fusion' ---- Fusion images

'train' ---- train dataset

'test' ---- test dataset

'detected_result' ---- predicted results

'Evaluation_result' ---- evaluation results

'Paper_figure' ---- Paper Illustrations
