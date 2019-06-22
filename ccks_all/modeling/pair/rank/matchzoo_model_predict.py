import csv
import json

import dill as dill
import pandas as pd

import matchzoo as mz
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from matchzoo import Embedding, load_model
from tqdm import tqdm

from ccks_all.cut_text import all_text_dic, kb_all_text_dic


def load_data(data_type, line_nub=-1):

    id_set = set()

    X_left = []
    X_left_id = []
    X_right = []
    X_right_id = []

    y = []

    json_line_s = open(root_dir.format(data_type + ".json.jieba.pre.json"), "r", encoding="utf-8").readlines()
    query_data_loder = tqdm(json_line_s)
    query_data_loder.set_description("load query data lines")
    for json_line in query_data_loder:
        tdata = json.loads(json_line)
        text_id = tdata["text_id"]
        query_text = all_text_dic[text_id]
        mention_data = tdata["mention_data"]
        for mention in mention_data:

            text_id_subject = text_id + mention["mention"]
            kb_id = mention["kb_id"]
            doc_text = kb_all_text_dic[kb_id]

            y_label = int(mention["label"])

            pid = text_id + "_" + kb_id

            if len(id_set) == line_nub:
                break

            if pid not in id_set:
                id_set.add(pid)

                X_left.append(query_text)
                X_right.append(doc_text)
                X_left_id.append(text_id_subject)
                X_right_id.append(kb_id)
                y.append(y_label)

        else:
            continue
        break

    df = pd.DataFrame({
        'text_left': X_left,
        'text_right': X_right,
        'id_left': X_left_id,
        'id_right': X_right_id,
        'label': y
    })
    return mz.pack(df)


root_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\{}"

print('data loading ...')
# train_pack_raw = load_data('train', 100000)
# dev_pack_raw = load_data('validate', 200)
test_pack_raw = load_data('test', 200)

model_path = r"D:/data/biendata/ccks2019_el/entityrank/m0/model/"
preprocess_path = model_path + "preprocessor.dill"
model = load_model(model_path)
preprocessor = dill.load(open(preprocess_path, "rb"))

# train_pack_processed = preprocessor.fit_transform(train_pack_raw)
# dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)

test_x, test_y = test_pack_processed.unpack()


pre = model.predict(test_x, 128)


pass
