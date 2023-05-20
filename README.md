# Pre-training with Machine Reader (PMR)
- This repo contains the code, the pre-trained checkpoints and the pre-training data for the following papers
  -  [From Cloze to Comprehension: Retrofitting Pre-trained Language Models to Pre-trained Machine Readers](https://arxiv.org/pdf/2212.04755.pdf), arXiv:2212.04755.
  -  [mPMR: A Multilingual Pre-trained Machine Reader at Scale](), ACL 2023.

## Requirements
* python 3.6.12
* pytorch 1.9.0+cu111 (downloaded from the official webpage)
* transformers 4.17.0
* tensorboardX 1.8
* tqdm 4.63.0
* sentencepiece==0.1.96
* Levenshtein
* rank_bm25
* tensorboardX

## Preprocessing Wikipedia
We downloaded the English Wikipedia dump (2022-01-01) from [Wikimedia](https://dumps.wikimedia.org/enwiki) and preprocessed it with [WikiExtractor](https://github.com/attardi/wikiextractor).
The following code is our command to extract plain text with hyperlinks from the Wiki dump, which will create a folder named ``en`` that stores all extracted pages.
```     
python WikiExtractor.py enwiki-20220801-pages-articles.xml.bz2 -c -l --json -o en   
```
Then we use the following code to do tokenization for all the wiki pages. The outputs are two file named ``en/processed/e2p.bz2`` and ``en/processed/e2c.bz2``.
``e2p`` is a dictionary that maps the entity (e) to its tokenized wiki page (p). ``e2c`` is another dictionary that maps the entity (e) to its anchor's context (c). 
```
cp -r wikiextractor/en ./
python wiki_preprocess_multiprocess.py --file en --processes 20
```
Once we get the tokenized data, we do sampling to select a number of context for each wiki entity. We set an upper bound (``--up``) of context number and
a filtering threshold (``--bottom``) to filter out the entities that have fewer contexts.
```     
python wiki_sampling.py --method fre --fre 10 --k 10 --file en --sample_data full-random10
```
We also generate the token id using the model tokenizer and save it in a cached file for fast reusage and accelerating the training process (``--evaluate`` for generating the cache file of test set).
```     
python wikipedia2mrc_multiprocess.py --processes 20 --file en --sample_data full-random10 --model_type roberta --model_name_or_path roberta-base --do_negative --buffer 7000000 --simple
```


## Pre-training
Once we get the cached file, we can start pre-training. Please refer to the following codes for pre-training base/large/xxlarge-sized models.
```
bash train.sh
bash train-large.sh
bash train-xxlarge.sh ## Notes: on top of ALBERT-xxlarge
```


## Fine-tuning 
* The downstream datasets are provided in ```$task/Data/$dataset```.
* Using the provided scripts to fine-tune a task-specific model. (Following is an exmaple of fine-tuning CONLL03)
  ```
  cd NER
  bash conll03.sh
  ```


## Citation
If the code is used in your research, please star our repo and cite our paper as follows:
```
@article{xu2022clozing,
  title={From Clozing to Comprehending: Retrofitting Pre-trained Language Model to Pre-trained Machine Reader},
  author={Xu, Weiwen and Li, Xin and Zhang, Wenxuan and Zhou, Meng and Bing, Lidong and Lam, Wai and Si, Luo},
  journal={arXiv preprint arXiv:2212.04755},
  year={2022}
}
```
     
