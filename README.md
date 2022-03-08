This repo contains the code and data for causality extraction task.

### Requirements ###

1) python 3.6
2) pytorch 1.7.1
3) CUDA 10.1
4) Transformers 3.5.0

### Dataset ###

https://drive.google.com/drive/folders/1DGmljtCkvNY2PUzZH94G74agPNyMoKPi?usp=sharing

5 fold splits for FinCausal2020 and FinCausal2021 is given in the above location. Use data_prep.py to convert them into the proper format for our code.

python3.6 data_prep.py in_file_csv out_file_json out_bert_file_json bert_tokenizer_name

in_file_csv: csv file in the splits

out_file_json: intermediate non-beat json file

out_bert_file_json: json file with BERT tokens. This file is used by cepn.py to train and test the model.

bert_tokenizer_name: bert-base-cased or bert-large-cased

### How to run ###

CEPN_Base

	python3.6 cepn.py FinCausal2020 config.ini target_dir train5fold

	python3.6 cepn.py FinCausal2021 config.ini target_dir train5fold

CEPN_Large

	python3.6 cepn.py FinCausal2020 config_large.ini target_dir train5fold

	python3.6 cepn.py FinCausal2021 config_large.ini target_dir train5fold

target_dir: Some directory where model and other output files are saved.

### Publication ###

If you use the source code or models from this work, please cite our paper:

@inproceedings{nayak2022cepn,

  author    = {Tapas Nayak, Soumya Sharma, Yash Butala, Koustuv Dasgupta, Pawan Goyal, and Niloy Ganguly},

  title     = {A Generative Approach for Financial Causality Extraction},

  booktitle = {Proceedings of The 2nd Workshop on Financial Technology on the Web (FinWeb)},

  year      = {2022}

}
