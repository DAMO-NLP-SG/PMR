import torch
from tqdm import tqdm
import argparse
import os
import logging
import math
logging.getLogger().setLevel(logging.INFO)


class MRCFeatures:
    '''
    MRC features
    '''
    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        start_positions,
        end_positions,
        doc_offset,
        len_query,
        len_context,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.doc_offset = doc_offset
        self.len_query = len_query
        self.len_context = len_context




def mix(args):
    langs = args.langs.split('_')
    evaluate = args.evaluate
    lang_count = {}
    cached_features_prefix = os.path.join(
        "cached_{}_{}_{}_{}_{}_{}".format(
            "PMR",
            "train" if not evaluate else "test",
            list(filter(None, args.model_type.split("/"))).pop(),
            str(args.max_query_length),
            str(args.max_seq_length),
            args.sample_data,
        ),
    )
    cached_features_prefix_cl = cached_features_prefix
    if args.do_negative:
        cached_features_prefix = cached_features_prefix + "_negative"
        cached_features_prefix_cl = cached_features_prefix_cl + "_negative"
        
    for lang in langs:
        logging.info('count the number of {} MRC example'.format(lang))
        cache_files = os.listdir(os.path.join(lang, "processed"))
        if lang == 'en':
            cache_files = [x for x in cache_files if x.startswith(cached_features_prefix_cl)]
        else:
            cache_files = [x for x in cache_files if x.startswith(cached_features_prefix)]
        cache_files = sorted(cache_files, key=lambda x: int(x.split("_")[-1]))
        features = torch.load(os.path.join(lang, "processed", cache_files[-1]))
        if len(cache_files) == 1:
            lang_count[lang] = len(features)
        else:
            lang_count[lang] = len(features) + (len(cache_files) - 1) * args.saved_buffer
        del features
    logging.info('the number of MRC examples for each language are {}'.format(str(lang_count)))

    # lang_count = {'ar': 2020502, 'bn': 410634, 'de': 14795826, 'el': 946114, 'en': 19303940, 'es': 7044972, 'fi': 1960636, 'fr': 10164216, 'hi': 242078, 'id': 1164662, 'it': 6421850, 'ja': 7338308, 'ko': 1597076, 'nl': 4185913, 'pl': 4765015, 'pt': 3648603, 'ru': 7342472, 'sv': 2808214, 'sw': 65724, 'te': 170664, 'th': 522434, 'tr': 1175276, 'vi': 1147772, 'zh': 4438004}


    num_total = sum(lang_count.values())
    buffer_num = math.ceil(num_total / args.buffer)
    logging.info('We need to mix all language data into several cache files with the buffer size of {}. All the cache files maintain the same language distribution as above'.format(args.buffer))
    logging.info("Total number of MRC examples = {}".format(num_total))
    logging.info("Number of MRC examples in each cache file = {}".format(args.buffer))
    logging.info("Number of cache files = {}".format(buffer_num))

    for lang in langs:
        save_dir = os.path.join(lang, "processed", args.out_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

            logging.info('allocate {} MRC example into multiple cache file'.format(lang))
            cache_files = os.listdir(os.path.join(lang, "processed"))
            if lang == 'en':
                save_path = os.path.join(save_dir, cached_features_prefix_cl)
                cache_files = [x for x in cache_files if x.startswith(cached_features_prefix_cl)]
            else:
                save_path = os.path.join(save_dir, cached_features_prefix)
                cache_files = [x for x in cache_files if x.startswith(cached_features_prefix)]
            cache_files = sorted(cache_files, key=lambda x: int(x.split("_")[-1]))
            num_per_cache = int(lang_count[lang] / num_total * args.buffer)
            i_buffer = 0
            mix_features = []
            for file in cache_files:
                features = torch.load(os.path.join(lang, "processed", file))
                keys = list(features.keys())
                for key in keys:
                    if len(mix_features) == num_per_cache:
                        save_path_i = save_path + "_" + str(i_buffer)
                        out_dict = {"{}".format(i_t): feature for i_t, feature in enumerate(mix_features)}
                        torch.save(out_dict, save_path_i)
                        del mix_features
                        del out_dict
                        i_buffer += 1
                        mix_features = []
                    feature_one = features.pop(key)
                    mix_features.append(feature_one)
            if len(mix_features) != 0:
                save_path_i = save_path + "_" + str(i_buffer)
                out_dict = {"{}".format(i_t): feature for i_t, feature in enumerate(mix_features)}
                torch.save(out_dict, save_path_i)
                del mix_features
                del out_dict
                i_buffer += 1
                mix_features = []
            assert i_buffer == buffer_num
    logging.info("allocate features Done!")

    for i_buffer in range(buffer_num):
        logging.info('mix MRC example into {}-th mixed file'.format(i_buffer))
        mix_features = {}
        offset = 0
        for lang in langs:
            load_dir = os.path.join(lang, "processed", args.out_dir)
            load_path_i = os.path.join(load_dir, cached_features_prefix + "_" + str(i_buffer))
            features_lang = torch.load(load_path_i)
            mix_features.update({"{}".format(str(int(key) + offset)): features_lang[key] for key in features_lang})
            offset = len(mix_features)
            del features_lang
        save_dir = os.path.join(args.out_dir, "processed")
        save_path = os.path.join(save_dir, cached_features_prefix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path_i = save_path + "_" + str(i_buffer)
        torch.save(mix_features, save_path_i)
        del mix_features
    logging.info("mix features Done!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", type=str, default="en_de_fr_ja_ru_es_it_zh_pt_ar",
                        help="wikipedia dump file directory")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.",)
    parser.add_argument("--max_query_length", type=int, default=128,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.",)
    parser.add_argument("--model_type", type=str, default="xlmr",
                        help="Model type")
    parser.add_argument("--buffer", type=int, default=3000000,
                        help="generate test set with size of PMR_test",)
    parser.add_argument("--saved_buffer", type=int, default=7000000,
                        help="generate test set with size of PMR_test",)
    parser.add_argument("--out_dir", type=str, default='mix',
                        help="the directory to save mix file", )
    parser.add_argument("--sample_data", type=str, default='full-random-10', help="the sampled file file for generating data")
    parser.add_argument("--evaluate", action="store_true",
                        help="generate test or train set.")
    parser.add_argument("--do_negative", action="store_true",
                        help="if add negative examples.")
    args = parser.parse_args()
    mix(args)