from datasets import load_dataset, load_metric
from transformers import Trainer
# dataset = load_dataset("glue", "cola")
# dataset = load_dataset("glue", "sst2")
# dataset = load_dataset("glue", "mrpc")
# dataset = load_dataset("glue", "stsb")
# dataset = load_dataset("glue", "qqp")
dataset = load_dataset("glue", "mnli")
print('get')
# dataset = load_dataset("glue", "qnli")
# dataset = load_dataset("glue", "rte")
# dataset = load_dataset("glue", "wnli")
# dataset = load_metric("super_glue", "boolq")
# dataset = load_metric("super_glue", "cb")
# dataset = load_metric("super_glue", "copa")
# dataset = load_metric("super_glue", "rte")
# dataset = load_metric("super_glue", "wic")
# dataset = load_metric("super_glue", "wsc")
# dataset = load_metric("super_glue", "multirc")
# dataset = load_metric("super_glue", "record")

# dataset = load_dataset("super_glue", "boolq")
# dataset.save_to_disk('./Data/boolq')
# dataset = load_dataset("super_glue", "cb")
# dataset.save_to_disk('./Data/cb')
# dataset = load_dataset("super_glue", "copa")
# dataset.save_to_disk('./Data/copa')
# dataset = load_dataset("super_glue", "rte")
# dataset.save_to_disk('./Data/rte')
# dataset = load_dataset("super_glue", "wic")
# dataset.save_to_disk('./Data/wic')
# dataset = load_dataset("super_glue", "wsc")
# dataset.save_to_disk('./Data/wsc')
# dataset = load_dataset("super_glue", "multirc")
# dataset.save_to_disk('./Data/multirc')
# dataset = load_dataset("super_glue", "record")
# dataset.save_to_disk('./Data/record')
# you can use any of the following config names as a second argument:
"ax", "cola", "mnli", "mnli_matched",
"mnli_mismatched", "mrpc", "qnli", "qqp",
"rte", "sst2", "stsb", "wnli"
['boolq', "cb", "copa", "rte", "wic", "wsc", "multirc", "record"]