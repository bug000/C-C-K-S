import pandas as pd

from ccks_ner.modeling.kbn_pair_model.clf_with_knowledge import get_data_generator, load_vocab, get_data_all

load_vocab()
# train_data_generator = get_data_generator("train")
datas = get_data_all("train", 100)

for d in datas:
    print(d)
