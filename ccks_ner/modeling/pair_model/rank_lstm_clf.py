import pickle
import random

import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, \
    BatchNormalization, Lambda
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, SpatialDropout2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

import matchzoo as mz

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# embedding_path = "D:/data/word2vec/zh/test.txt"
embedding_path = "D:/data/word2vec/zh/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.utf8.txt"

# train_dir = r"C:\python3workspace\kera_ner_demo\ccks_ner\modeling\pair_model\dt\m3\{}"
train_dir = r"D:\data\biendata\ccks2019_el\entityclf\m8\{}"
log_filepath = train_dir.format(r"log")
toka_path = train_dir.format(r"\toka.bin")
model_path = train_dir.format(r"bilstm_model.hdf5")

root_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\clf_data\split\{}"
train = pd.read_csv(root_path.format("train.tsv.hanlp.tsv"), sep="\t", nrows=500, header=None)
test = pd.read_csv(root_path.format("test.tsv.hanlp.tsv"), sep="\t", nrows=200, header=None)
# val = pd.read_csv(root_path.format("validate.tsv.hanlp.tsv"), sep="\t", nrows=100, header=None)

full_text = list(train.iloc[:, 3].values) + list(train.iloc[:, 4].values) + list(test.iloc[:, 3].values) + list(
    test.iloc[:, 4].values)

s = train.iloc[:, 4].values

y = train.iloc[:, 0]
y_test = test.iloc[:, 0]

tk = Tokenizer(lower=True, filters='')
full_text = [str(s) for s in full_text]
tk.fit_on_texts(full_text)

train_a_tokenized = tk.texts_to_sequences(train.iloc[:, 3].values)
train_b_tokenized = tk.texts_to_sequences([str(s) for s in train.iloc[:, 4].values])
test_a_tokenized = tk.texts_to_sequences(test.iloc[:, 3].values)
test_b_tokenized = tk.texts_to_sequences(test.iloc[:, 4].values)

pickle.dump(tk, open(toka_path, 'wb'))

max_len_a = 30
max_len_b = 30
X_train_a = pad_sequences(train_a_tokenized, maxlen=max_len_a)
X_train_b = pad_sequences(train_b_tokenized, maxlen=max_len_b)
X_test_a = pad_sequences(test_a_tokenized, maxlen=max_len_a)
X_test_b = pad_sequences(test_b_tokenized, maxlen=max_len_b)
embed_size = 300
max_v_len = 10000


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding="utf-8"))

word_index = tk.word_index
nb_words = min(max_v_len, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_v_len: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(y.values.reshape(-1, 1))

"""call back"""
check_point = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
# early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
tb_cb = TensorBoard(log_dir=log_filepath)


def _kernel_layer(mu: float, sigma: float) -> keras.layers.Layer:
    """
    Gaussian kernel layer in KNRM.

    :param mu: Float, mean of the kernel.
    :param sigma: Float, sigma of the kernel.
    :return: `keras.layers.Layer`.
    """

    def kernel(x):
        return K.tf.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)

    return keras.layers.Activation(kernel)


def build_model(_params):
    inp_a = Input(shape=(max_len_a,))
    inp_b = Input(shape=(max_len_b,))
    q_embed = Embedding(max_v_len+1, embed_size, weights=[embedding_matrix], trainable=False)(inp_a)
    d_embed = Embedding(max_v_len+1, embed_size, weights=[embedding_matrix], trainable=False)(inp_b)

    q_convs = []
    d_convs = []
    for i in range(_params['max_ngram']):
        c = keras.layers.Conv1D(
            _params['filters'], i + 1,
            activation=_params['conv_activation_func'],
            padding='same'
        )
        q_convs.append(c(q_embed))
        d_convs.append(c(d_embed))

    KM = []
    for qi in range(_params['max_ngram']):
        for di in range(_params['max_ngram']):
            # do not match n-gram with different length if use crossmatch
            if not _params['use_crossmatch'] and qi != di:
                continue
            q_ngram = q_convs[qi]
            d_ngram = d_convs[di]
            mm = keras.layers.Dot(axes=[2, 2],
                                  normalize=True)([q_ngram, d_ngram])

            for i in range(_params['kernel_num']):
                mu = 1. / (_params['kernel_num'] - 1) + (2. * i) / (
                        _params['kernel_num'] - 1) - 1.0
                sigma = _params['sigma']
                if mu > 1.0:
                    sigma = _params['exact_sigma']
                    mu = 1.0
                mm_exp = _kernel_layer(mu, sigma)(mm)
                mm_doc_sum = keras.layers.Lambda(
                    lambda x: K.tf.reduce_sum(x, 2))(
                    mm_exp)
                mm_log = keras.layers.Activation(K.tf.log1p)(mm_doc_sum)
                mm_sum = keras.layers.Lambda(
                    lambda x: K.tf.reduce_sum(x, 1))(mm_log)
                KM.append(mm_sum)

    phi = Lambda(lambda x: K.tf.stack(x, 1))(KM)
    out = keras.layers.Dense(1, activation='linear')(phi)
    """:fine-tune"""
    model = Model(inputs=[inp_a, inp_b], outputs=out)
    model.trainable = True
    for layer in model.layers[:1]:
        layer.trainable = False
    model.summary()

    """:train"""
    loss = mz.losses.RankCrossEntropyLoss(num_neg=1)
    optimizer = 'adadelta'
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    # model.fit_generator
    model.fit([X_train_a, X_train_b], y_ohe,
              batch_size=24,
              epochs=20,
              validation_split=0.3,
              verbose=1,
              # class_weight={0: 1, 1: 20},
              callbacks=[check_point, early_stop, tb_cb])
    K.clear_session()
    tf.reset_default_graph()

    model = load_model(model_path)
    return model


_params = {
    "max_ngram": 3,
    "conv_activation_func": 'tanh',
    "filters": 128,
    "kernel_num": 11,
    "sigma": 0.1,
    "exact_sigma": 0.001,
    "use_crossmatch": True,
}

td_model = build_model(_params)

pred = td_model.predict([X_test_a, X_test_b], batch_size=1024)
predictions = np.round(np.argmax(pred, axis=1)).astype(int)

report = classification_report(y_test, predictions)

print(report)

"""

no embedding m4

              precision    recall  f1-score   support

           0       0.97      0.48      0.65    179485
           1       0.27      0.94      0.42     36257

    accuracy                           0.56    215742
   macro avg       0.62      0.71      0.53    215742
weighted avg       0.86      0.56      0.61    215742


embedding m5
              precision    recall  f1-score   support

           0       0.99      0.78      0.87    179485
           1       0.47      0.94      0.63     36257

    accuracy                           0.81    215742
   macro avg       0.73      0.86      0.75    215742
weighted avg       0.90      0.81      0.83    215742



m6
              precision    recall  f1-score   support

           0       0.98      0.83      0.90    179485
           1       0.52      0.93      0.67     36257

    accuracy                           0.84    215742
   macro avg       0.75      0.88      0.78    215742
weighted avg       0.91      0.84      0.86    215742



"""
