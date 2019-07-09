import json
import pickle
import random
import re

import dill
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Lambda

from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, get_custom_objects
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
from tqdm import tqdm

from ccks_all.cut_text import train_text_dic, all_text_dic, kb_all_text_dic, load_cut_text

from keras import backend as K

from ccks_all.modeling.bertmodel.preprocess import PairTokenizer, BertPreProcess, Preprocess, BertNerProcess
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
model_dir = r"D:\data\biendata\ccks2019_el\entityclf\m21\{}"
log_filepath = model_dir.format(r"log")
model_path = model_dir.format(r"best_model.hdf5")

token_dict = {}
with open(dict_path, 'r', encoding='utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class DataGener(object):

    def __init__(self, data_type, processer: BertNerProcess, batch_size=32):
        self.json_line_s = open(root_dir.format(data_type), "r", encoding="utf-8").readlines()
        self.processer = processer
        self.batch_size = batch_size

    def __len__(self):
        l_nub = len(self.json_line_s)
        steps = l_nub // self.batch_size
        if l_nub % self.batch_size != 0:
            steps += 1
        return steps

    def __iter__(self):
        while True:

            X1, X2 = [], []
            y1, y2 = [], []

            for i, json_line in enumerate(self.json_line_s):
                tdata = json.loads(json_line)
                query_text = tdata["text"]

                mention_data = tdata["mention_data"]
                for mention in mention_data:
                    offset = mention["offset"]
                    mention = mention["mention"]
                    end_offset = offset + len(mention)

                    x1, x2 = self.processer.process_line(query_text)
                    X1.append(x1)
                    X2.append(x2)
                    y1.append(offset)
                    y2.append(end_offset)

                    if len(y1) == self.batch_size or i == (len(self.json_line_s)-1):
                        batch_data = [self.processer.seq_padding(X1.copy()),
                                      self.processer.seq_padding(X2.copy()),
                                      self.processer.seq_padding(y1.copy()),
                                      self.processer.seq_padding(y2.copy()),
                                      ]
                        X1 = []
                        X2 = []
                        y1 = []
                        y2 = []
                        yield batch_data, None

    def get_bert_pair_text_all(self, max_len=-1):
        X1, X2 = [], []
        y1, y2 = [], []

        ohe = OneHotEncoder(sparse=False, categories='auto')
        ohe.fit(np.asarray([0, 1]).reshape(-1, 1))
        dload = tqdm(self.json_line_s[:max_len])
        dload.set_description("load data .")
        for json_line in dload:
            tdata = json.loads(json_line)
            query_text = tdata["text"]

            mention_data = tdata["mention_data"]
            for i, mention in enumerate(mention_data):
                offset = mention["offset"]
                mention = mention["mention"]
                end_offset = offset + len(mention)

                x1, x2 = self.processer.process_line(query_text)
                X1.append(x1)
                X2.append(x2)
                y1.append(offset)
                y2.append(end_offset)

        all_data = [self.processer.seq_padding(X1.copy()),
                      self.processer.seq_padding(X2.copy()),
                      self.processer.seq_padding(y1.copy()),
                      self.processer.seq_padding(y2.copy()),
                      ]

        return all_data, None


class Evaluate(Callback):
    def __init__(self, tokenizer, dev_data_x, lr, mlr):
        super().__init__()
        self.ACC = []
        self.best = 0.
        self.passed = 0
        self.tokenizer = tokenizer
        self.dev_data_x = dev_data_x
        self.learning_rate = lr
        self.min_learning_rate = mlr

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * self.learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (self.learning_rate - self.min_learning_rate)
            lr += self.min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            self.model.save_weights(model_path)
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))

    def evaluate(self):
        A = 1e-10
        F = open('dev_pred.json', 'w')
        for d in tqdm(iter(dev_data)):
            R = self.extract_entity(d[0], d[1])
            if R == d[2]:
                A += 1
            s = ', '.join(d + (R,))
            F.write(s.encode('utf-8') + '\n')
        F.close()
        return A / len(dev_data)


def get_layer(x1, x2):
    for l in bert_model.layers:
        l.trainable = True

    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)

    x = bert_model([x1, x2])
    ps1 = Dense(1, use_bias=False)(x)
    ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
    ps2 = Dense(1, use_bias=False)(x)
    ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

    return ps1, ps2


def build_model(lr: float, lr_d: float, process: BertNerProcess):
    """data"""
    # validate train
    train_gener = DataGener("t.json", processer=process, batch_size=32)
    val_gener = DataGener("t.json", processer=process, batch_size=64)

    """layers"""
    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入
    s1_in = Input(shape=(None,))  # 实体左边界（标签）
    s2_in = Input(shape=(None,))  # 实体右边界（标签）
    p1, p2 = get_layer(x1_in, x2_in)

    """call back"""
    check_point = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    tb_cb = TensorBoard(log_dir=log_filepath)

    """fine-tune"""
    model = Model(inputs=[x1_in, x2_in], outputs=[p1, p2])
    # model.trainable = True
    # for layer in model.layers[:1]:
    #     layer.trainable = False
    model.summary()

    """train"""
    # vald = val_gener.get_bert_pair_text_all()
    # trnd = train_gener.get_bert_pair_text_all()
    # model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy", f1])
    loss1 = K.mean(K.categorical_crossentropy(s1_in, p1, from_logits=True))
    p2 -= (1 - K.cumsum(s1_in, 1)) * 1e10
    loss2 = K.mean(K.categorical_crossentropy(s2_in, p2, from_logits=True))
    loss = loss1 + loss2

    model.add_loss(loss)
    model.compile(optimizer=Adam(lr=lr, decay=lr_d))
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
                        callbacks=[check_point, early_stop, tb_cb])

    model.save(model_path)
    K.clear_session()
    tf.reset_default_graph()

    model = load_model(model_path, custom_objects=get_custom_objects())
    return model




def main():
    tokenizer = PairTokenizer(token_dict)
    process = BertNerProcess(tokenizer)
    dill.dump(process, open(model_dir.format(r"process.dill"), "wb"))
    # build_vocab(toka_path)
    # td_model = build_model(lr=1e-4, lr_d=0)
    td_model = build_model(lr=1e-5, lr_d=1e-7, process=process)


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
