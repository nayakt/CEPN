This repo contains the code and data for causality extraction task.

### Requirements ###

1) python 3.6
2) pytorch 1.7.1
3) CUDA 10.1
4) Transformers 3.5.0

### How to run ###

CEPN_Base

	python3.6 cepn.py SemEval config.ini target_dir train

	python3.6 cepn.py FinCausal2020 config.ini target_dir train5fold

	python3.6 cepn.py FinCausal2021 config.ini target_dir train5fold

CEPN_Large

	python3.6 cepn.py SemEval config_large.ini target_dir train

	python3.6 cepn.py FinCausal2020 config_large.ini target_dir train5fold

	python3.6 cepn.py FinCausal2021 config_large.ini target_dir train5fold

target_dir: Some directory where model and other output files are saved.