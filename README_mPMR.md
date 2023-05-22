# mPMR
mPMR: A Multilingual Pre-trained Machine Reader at Scale

## Requirements
* python 3.6.12
* torch 1.9.0+cu111 (downloaded from the official webpage)
* transformers 4.17.0
* tensorboardX 1.8
* tqdm 4.63.0
* sentencepiece==0.1.96
* opencc
* Levenshtein
* spacy==3.4.1
* sacremoses
* pythainlp

## Preprocessing Wikipedia (English as an example)
1. We downloaded the English Wikipedia dump (2022-01-01) from [Wikimedia](https://dumps.wikimedia.org/enwiki) and preprocessed it with [WikiExtractor](https://github.com/attardi/wikiextractor).
The following code is our command to extract plain text with hyperlinks from the Wiki dump, which will create a folder named ``en`` that stores all extracted pages.
    ```     
    python WikiExtractor.py enwiki-20220801-pages-articles.xml.bz2 -c -l --json -o en   
    cp -r ./en ../mPMR/
    ```
2. Then we use the following code to do tokenization for all the wiki pages. The outputs are two file named ``en/processed/e2p.bz2`` and ``en/processed/e2c.bz2``.
``e2p`` is a dictionary that maps the entity (e) to its tokenized wiki page (p). ``e2c`` is another dictionary that maps the entity (e) to its anchor's context (c). 
    ```
    python reprocessing_mPMR/wiki_preprocess_multiprocess.py --file en --processes 20
    ```
3. Once we get the tokenized data, we do sampling to select a number of context for each wiki entity. We set an upper bound (``--up``) of context number and
a filtering threshold (``--bottom``) to filter out the entities that have fewer contexts.
    ```     
    python reprocessing_mPMR/wiki_sampling.py --bottom 10 --up 10 --file en --sample_data full-random-10
    ```
4. We also generate the token id using the model tokenizer and save it in a cached file for fast reusage and accelerating the training process (``--evaluate`` for generating the cache file of test set). 
    ```     
    python reprocessing_mPMR/wikipedia2mrc_multiprocess.py --processes 20 --file en --sample_data full-random-10 --model_type xlmr --model_name_or_path xlm-roberta-base --do_negative --buffer 7000000 --simple --window 100
    python reprocessing_mPMR/wikipedia2mrc_multiprocess.py --processes 20 --file en --sample_data full-random-10 --model_type xlmr --model_name_or_path xlm-roberta-base --do_negative --buffer 7000000 --simple --window 100 --evaluate
    ```


## Preprocessing Wikipedia (all languages)
After extracting wikipedia pages for all languages (step 1 in the last section), we provide preprocessing scripts to create MRC examples for all languages. 
```
bash reprocessing_mPMR/wiki_preprocess.sh
bash reprocessing_mPMR/wiki_sampling.sh
bash reprocessing_mPMR/wiki2mrc.sh
```

## Mix all training data together
Since each cache file contains the MRC examples of the same language, we do another mix procedure such that each cache file contains exmaples from all languages with the same distribution (``--evaluate`` for mixing the cache file of test set). 
   ```     
   python preprocessing_mPMR/wiki_mix_fast.py --langs ar_bn_de_fi_fr_el_en_es_hi_id_it_ja_ko_nl_pl_pt_ru_sv_sw_te_th_tr_vi_zh --sample_data full-random-10 --model_type xlmr --saved_buffer 7000000 --buffer 10000000 --do_negative --out_dir mix24
   ```

## Pre-training
Once we get the cached file, we can start pre-training. Please refer to the following codes for pre-training base/large-sized models.
```
bash mtrain.sh
bash mtrain-large.sh
```


## Fine-tuning 
* The downstream datasets are provided in ```xxx```. Please download them and put them in ```$task/Data/$dataset```. (e.g. CONLL03 dataset would be placed in ```NER/Data/conll03```)
* Using the provided scripts to fine-tune a task-specific model. (Following is an exmaple of fine-tuning CONLL03)
  ```
  cd NER
  bash conll.sh
  ```
