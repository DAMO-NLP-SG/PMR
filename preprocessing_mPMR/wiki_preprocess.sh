for lang in ar bn el de 'fi' en es hi id it ja ko nl zh pl pt sw te fr sv tr ru vi th
do
python wiki_preprocess_multiprocess.py --processes 20 --file $lang
done