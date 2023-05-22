for lang in ar bn el de 'fi' es hi id it ja ko nl zh pl pt sw te fr sv tr ru vi th
do
python wikipedia2mrc_multiprocess.py --processes 20 --file $lang --sample_data full-random-10 --model_type xlmr --model_name_or_path xlm-roberta-base --do_negative --buffer 7000000 --simple --window 100 --evaluate
python wikipedia2mrc_multiprocess.py --processes 20 --file $lang --sample_data full-random-10 --model_type xlmr --model_name_or_path xlm-roberta-base --do_negative --buffer 7000000 --simple --window 100
done
python wikipedia2mrc_multiprocess.py --processes 20 --file en --sample_data full-random-10 --model_type xlmr --model_name_or_path xlm-roberta-base --do_negative --buffer 7000000 --simple --window 100 --evaluate
python wikipedia2mrc_multiprocess.py --processes 20 --file en --sample_data full-random-10 --model_type xlmr --model_name_or_path xlm-roberta-base --do_negative --buffer 7000000 --simple --window 100