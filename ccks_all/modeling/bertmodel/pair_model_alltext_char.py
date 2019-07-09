import json
import pickle
import random

import dill
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, Lambda

from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, get_custom_objects
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
from tqdm import tqdm

from ccks_all.cut_text import train_text_dic, all_text_dic, kb_all_text_dic, load_cut_text

from keras import backend as K

from ccks_all.modeling.bertmodel.preprocess import PairTokenizer, BertPreProcess
from ccks_all.modeling.utils import Metrics

max_q = 30
max_d = 450

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

root_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\{}"

bert_dir = r'D:\data\bert\chinese-bert_chinese_wwm_L-12_H-768_A-12'

config_path = bert_dir + r"\bert_config.json"
checkpoint_path = bert_dir + r"\bert_model.ckpt"
dict_path = bert_dir + r'\vocab.txt'
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

# train_dir = r"C:\python3workspace\kera_ner_demo\ccks_ner\modeling\pair_model\dt\m3\{}"
model_dir = r"D:\data\biendata\ccks2019_el\entityclf\m22\{}"
log_filepath = model_dir.format(r"log")
model_path = model_dir.format(r"best_model.hdf5")

token_dict = {}
with open(dict_path, 'r', encoding='utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

entity_text_dict = load_cut_text("kb_data.all.jieba.text.tsv", col=1)


class DataGener(object):

    def __init__(self, data_type, processer: BertPreProcess, batch_size=32, max_len=-1):
        self.json_line_s = open(root_dir.format(data_type), "r", encoding="utf-8").readlines()
        self.processer = processer
        self.batch_size = batch_size
        self.max_len = max_len

    def __len__(self):
        l_nub = len(self.json_line_s)
        if self.max_len != -1:
            l_nub = self.max_len
        steps = l_nub // self.batch_size
        if l_nub % self.batch_size != 0:
            steps += 1
        return steps

    def __iter__(self):
        while True:
            id_set = set()

            X1, X2 = [], []

            y = []

            for json_line in self.json_line_s:
                tdata = json.loads(json_line)
                text_id = tdata["text_id"]
                query_text = tdata["text"]

                mention_data = tdata["mention_data"]
                for i, mention in enumerate(mention_data):
                    kb_id = mention["kb_id"]
                    # 加载未分词的 doc_text
                    doc_text = entity_text_dict[kb_id]

                    y_label = int(mention["label"])

                    pid = text_id + "_" + kb_id

                    if pid not in id_set:
                        id_set.add(pid)

                        x1, x2 = self.processer.process_line(query_text, doc_text)
                        X1.append(x1)
                        X2.append(x2)
                        y.append(y_label)

                        if len(id_set) == self.max_len:
                            batch_X = [self.processer.seq_padding(X1.copy()), self.processer.seq_padding(X2.copy())]
                            batch_y = y.copy()
                            yield batch_X, batch_y
                            break

                        if len(y) == self.batch_size or i == mention_data[-1]:
                            batch_X = [self.processer.seq_padding(X1.copy()), self.processer.seq_padding(X2.copy())]
                            batch_y = y.copy()

                            X1 = []
                            X2 = []
                            y = []

                            yield batch_X, batch_y

                else:
                    continue
                break

    def get_bert_pair_text_all(self):
        id_set = set()

        X1, X2 = [], []

        y = []

        ohe = OneHotEncoder(sparse=False, categories='auto')
        ohe.fit(np.asarray([0, 1]).reshape(-1, 1))
        dload = tqdm(self.json_line_s)
        dload.set_description("load data .")
        for json_line in dload:
            tdata = json.loads(json_line)
            text_id = tdata["text_id"]
            query_text = tdata["text"]

            mention_data = tdata["mention_data"]
            for i, mention in enumerate(mention_data):
                kb_id = mention["kb_id"]
                # 加载未分词的 doc_text
                doc_text = entity_text_dict[kb_id]

                y_label = int(mention["label"])

                pid = text_id + "_" + kb_id

                if pid not in id_set:
                    id_set.add(pid)

                    x1, x2 = self.processer.process_line(query_text, doc_text)
                    X1.append(x1)
                    X2.append(x2)
                    y.append(y_label)

                if len(id_set) == self.max_len:
                    break
            else:
                continue
            break

        X = [self.processer.seq_padding(X1.copy()), self.processer.seq_padding(X2.copy())]

        return X, y


def get_layer(inp_a, inp_b):
    x = bert_model([inp_a, inp_b])
    for l in bert_model.layers[68:]:
        l.trainable = True
    x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
    p = Dense(1, activation='sigmoid')(x)
    return p


def build_model(lr: float, lr_d: float, process: BertPreProcess):
    """data"""

    train_gener = DataGener("train.json.crf.m30.CRFDropModel.expand.pre.json", processer=process, batch_size=8, max_len=-1)
    val_gener = DataGener("validate.json.crf.m30.CRFDropModel.expand.pre.json", processer=process, batch_size=16, max_len=-1)

    """layers"""
    inp_a = Input(shape=(None,))
    inp_b = Input(shape=(None,))
    out = get_layer(inp_a, inp_b)

    """call back"""
    check_point = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    tb_cb = TensorBoard(log_dir=log_filepath)
    metrics = Metrics()

    """fine-tune"""
    model = Model(inputs=[inp_a, inp_b], outputs=out)
    model.trainable = True
    # for layer in model.layers[:1]:
    #     layer.trainable = False
    model.summary()

    """train"""
    # vald = val_gener.get_bert_pair_text_all()
    # trnd = train_gener.get_bert_pair_text_all()
    # model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy", f1])
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    # model.fit(x=trnd[0],
    #           y=trnd[1],
    #           validation_data=vald,
    #           epochs=3,
    #           class_weight="auto",
    #           callbacks=[check_point, early_stop, tb_cb])
    model.fit_generator(train_gener.__iter__(),
                        steps_per_epoch=train_gener.__len__(),
                        epochs=5,
                        validation_data=val_gener.__iter__(),
                        validation_steps=val_gener.__len__(),
                        class_weight="auto",
                        callbacks=[check_point, early_stop, tb_cb, metrics])

    model.save(model_path)
    K.clear_session()
    tf.reset_default_graph()

    model = load_model(model_path, custom_objects=get_custom_objects())
    return model


def predict(model, tk):
    test_gener = DataGener("test.json.jieba.search.pre.json", tk)
    X, y = test_gener.get_bert_pair_text_all()
    pred = model.predict(X, batch_size=1024, verbose=1)

    predictions = np.round(np.argmax(pred, axis=1)).astype(int)
    y_test = np.round(np.argmax(y, axis=1)).astype(int)

    report = classification_report(y_test, predictions)

    print(report)


def main():
    tokenizer = PairTokenizer(token_dict)
    process = BertPreProcess(tokenizer)
    dill.dump(process, open(model_dir.format(r"process.dill"), "wb"))
    # build_vocab(toka_path)
    # td_model = build_model(lr=1e-4, lr_d=0)
    td_model = build_model(lr=1e-5, lr_d=1e-7, process=process)
    # td_model = get_model()
    predict(td_model, tk=tokenizer)


if __name__ == '__main__':
    main()

# pred = td_model.predict([X_test_a, X_test_b], batch_size=1024)
# predictions = np.round(np.argmax(pred, axis=1)).astype(int)
#
# report = classification_report(y_test, predictions)
#
# print(report)

"""


m17
              precision    recall  f1-score   support

           0       0.98      0.99      0.98      9630
           1       0.50      0.38      0.43       370

    accuracy                           0.96     10000
   macro avg       0.74      0.68      0.71     10000
weighted avg       0.96      0.96      0.96     10000




embedding m15
              precision    recall  f1-score   support

           0       0.97      0.99      0.98      9630
           1       0.49      0.32      0.39       370

    accuracy                           0.96     10000
   macro avg       0.73      0.65      0.68     10000
weighted avg       0.96      0.96      0.96     10000





"""
