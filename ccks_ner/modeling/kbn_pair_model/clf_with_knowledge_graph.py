import json
import pickle
import random

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

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

kb_path = "D:/data/biendata/ccks2019_el/ccks_train_data/kb_data.hanlp.txt"

# embedding_path = "D:/data/word2vec/zh/test.txt"
embedding_path = "D:/data/word2vec/zh/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.utf8.txt"

# train_dir = r"C:\python3workspace\kera_ner_demo\ccks_ner\modeling\pair_model\dt\m3\{}"
model_dir = r"D:\data\biendata\ccks2019_el\entityclf\m7\{}"
log_filepath = model_dir.format(r"log")
toka_path = model_dir.format(r"\toka.bin")
model_path = model_dir.format(r"bilstm_model.hdf5")

max_len_a = 50
max_len_object = 80

embed_size = 300
max_features = 20000

tk = Tokenizer(lower=True, filters='')
root_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\clf_data\split\{}"


def get_layer(inp_a, inp_type, inp_object_s):
    def load_embedding(toka):
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding="utf-8"))

        word_index = toka.word_index
        nb_words = min(max_features, len(word_index))
        embedding_matrix_ = np.zeros((nb_words + 1, embed_size))
        for word, i in word_index.items():
            if i >= max_features: continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None: embedding_matrix_[i] = embedding_vector
        return embedding_matrix_, nb_words

    embedding_matrix, nb_words = load_embedding(tk)
    embed_layer = Embedding(nb_words + 1, embed_size, weights=[embedding_matrix], trainable=False)

    x_a = embed_layer(inp_a)
    x_a = SpatialDropout1D(0.3)(x_a)

    kg_dropout_layer = SpatialDropout1D(0.3)
    kg_encoder_layer = Bidirectional(CuDNNLSTM(64, return_sequences=False))
    kg_vec = Lambda(lambda x: kg_encoder_layer(
        kg_dropout_layer(
            embed_layer(x)
        )
    ))(inp_object_s)

    avg_pool_a = GlobalAveragePooling1D()(x_a)
    max_pool_a = GlobalMaxPooling1D()(x_a)
    avg_pool_b = GlobalAveragePooling1D()(kg_vec)
    max_pool_b = GlobalMaxPooling1D()(kg_vec)

    x_a = concatenate([avg_pool_a, max_pool_a])
    x_a = BatchNormalization()(x_a)
    x_a = Dropout(0.3)(Dense(32, activation='relu')(x_a))

    x_b = concatenate([avg_pool_b, max_pool_b])
    x_b = BatchNormalization()(x_b)
    x_b = Dropout(0.1)(Dense(64, activation='relu')(x_b))

    x = concatenate([x_a, x_b, inp_type])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(64, activation='relu')(x))

    out = Dense(2, activation="sigmoid")(x)
    return out


def build_vocab():
    full_text = []

    kb_lines = open(kb_path, "r", encoding="utf-8").readlines()
    for line in kb_lines:
        kb_data_s = json.loads(line)
        for kb_data in kb_data_s:
            full_text.append(kb_data.values())

    train = pd.read_csv(root_path.format("train.tsv.hanlp.tsv"), sep="\t", nrows=50000, header=None)
    test = pd.read_csv(root_path.format("test.tsv.hanlp.tsv"), sep="\t", nrows=20000, header=None)
    # val = pd.read_csv(root_path.format("validate.tsv.hanlp.tsv"), sep="\t", nrows=100, header=None)

    full_text.append(list(train.iloc[:, 3].values))
    full_text.append(list(train.iloc[:, 4].values))
    full_text.append(list(test.iloc[:, 3].values))
    full_text.append(list(test.iloc[:, 4].values))

    full_text = [str(s) for s in full_text]
    tk.fit_on_texts(full_text)


def get_data_generator(data_type: str):
    entity_dict = {}
    for line in open(kb_path, "r", encoding="utf-8"):
        js_line = json.loads(line)
        entity_dict[js_line["subject_id"]] = js_line

    train = pd.read_csv(root_path.format(data_type + ".tsv.hanlp.tsv"), sep="\t", nrows=50000, header=None)

    y = train.iloc[:, 0]
    train_a_tokenized = tk.texts_to_sequences(train.iloc[:, 3].values)
    train_b_tokenized = tk.texts_to_sequences([str(s) for s in train.iloc[:, 4].values])

    X_train_a = pad_sequences(train_a_tokenized, maxlen=max_len_a)
    # X_train_b = pad_sequences(train_b_tokenized, maxlen=max_len_b)

    ohe = OneHotEncoder(sparse=False)
    yield ""


def build_model(lr=0.0, lr_d=0.0):
    """data"""
    build_vocab()
    pickle.dump(tk, open(toka_path, 'wb'))
    train_data_generator = get_data_generator("train")

    """layers"""
    inp_a = Input(shape=(max_len_a,))
    inp_type = Input(shape=(1,))
    inp_object_s = Input(shape=(10, max_len_object,))
    out = get_layer(inp_a, inp_type, inp_object_s)

    """call back"""
    check_point = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    # early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    tb_cb = TensorBoard(log_dir=log_filepath)

    """fine-tune"""
    model = Model(inputs=[inp_a, inp_type, inp_object_s], outputs=out)
    # model.trainable = True
    # for layer in model.layers[:1]:
    #     layer.trainable = False
    # model.summary()

    """train"""
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    model.fit_generator(train_data_generator,
                        batch_size=24,
                        epochs=20,
                        validation_split=0.3,
                        verbose=1,
                        class_weight={0: 1, 1: 10},
                        callbacks=[check_point, early_stop, tb_cb])
    K.clear_session()
    tf.reset_default_graph()

    model = load_model(model_path)
    return model


td_model = build_model(lr=1e-4, lr_d=0)

# pred = td_model.predict([X_test_a, X_test_b], batch_size=1024)
# predictions = np.round(np.argmax(pred, axis=1)).astype(int)
#
# report = classification_report(y_test, predictions)
#
# print(report)

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
