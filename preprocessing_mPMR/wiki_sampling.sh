for lang in ar bn el de 'fi' en es hi id it ja ko nl zh pl pt sw te fr sv tr ru vi th
do
python wiki_sampling.py --bottom 5 --up 10 --file $lang --sample_data full-random-10
done