import pickle

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, \
    BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, SpatialDropout2D
from keras.models import Model, load_model
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

embedding_path = "D:/data/word2vec/zh/test.txt"

train_dir = r"C:\python3workspace\kera_ner_demo\ccks_ner\modeling\pair_model\dt\m3\{}"
log_filepath = train_dir.format(r"log")
toka_path = train_dir.format(r"\toka.bin")
model_path = train_dir.format(r"bilstm_model.hdf5")

root_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\clf_data\split\{}"
train = pd.read_csv(root_path.format("train.tsv.hanlp.tsv"), sep="\t", nrows=30000, header=None)
test = pd.read_csv(root_path.format("test.tsv.hanlp.tsv"), sep="\t", nrows=20000, header=None)
# val = pd.read_csv(root_path.format("validate.tsv.hanlp.tsv"), sep="\t", nrows=100, header=None)

full_text = list(train.iloc[:, 3].values) + \
            list(train.iloc[:, 4].values) + \
            list(test.iloc[:, 3].values) + list(test.iloc[:, 4].values)

y = train.iloc[:, 0]
y_test = test.iloc[:, 0]

tk = Tokenizer(lower=True, filters='')
tk.fit_on_texts(full_text)

train_a_tokenized = tk.texts_to_sequences(train.iloc[:, 3].values)
train_b_tokenized = tk.texts_to_sequences(train.iloc[:, 4].values)
test_a_tokenized = tk.texts_to_sequences(test.iloc[:, 3].values)
test_b_tokenized = tk.texts_to_sequences(test.iloc[:, 4].values)

pickle.dump(tk, open(toka_path, 'wb'))

max_len_a = 50
max_len_b = 800
X_train_a = pad_sequences(train_a_tokenized, maxlen=max_len_a)
X_train_b = pad_sequences(train_b_tokenized, maxlen=max_len_b)
X_test_a = pad_sequences(test_a_tokenized, maxlen=max_len_a)
X_test_b = pad_sequences(test_b_tokenized, maxlen=max_len_b)
embed_size = 300
max_features = 20000


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding="utf-8"))

word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(y.values.reshape(-1, 1))

check_point = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)

tb_cb = TensorBoard(log_dir=log_filepath, write_images=False, histogram_freq=1)


def build_model(lr=0.0, lr_d=0.0):
    inp_a = Input(shape=(max_len_a,))
    inp_b = Input(shape=(max_len_b,))
    x_a = Embedding(nb_words + 1, embed_size, weights=[embedding_matrix], trainable=False)(inp_a)
    x_b = Embedding(nb_words + 1, embed_size, weights=[embedding_matrix], trainable=False)(inp_b)
    x_a = SpatialDropout1D(0.3)(x_a)
    x_b = SpatialDropout1D(0.3)(x_b)

    x_lstm_a = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x_a)
    x_lstm_b = Bidirectional(CuDNNLSTM(512, return_sequences=True))(x_b)
    xc_a = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_lstm_a)
    # xc_a = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_lstm_a)
    xc_b = Conv1D(128, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_lstm_b)
    # xc_b = Conv1D(128, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_lstm_b)
    avg_pool_a = GlobalAveragePooling1D()(xc_a)
    max_pool_a = GlobalMaxPooling1D()(xc_a)
    avg_pool_b = GlobalAveragePooling1D()(xc_b)
    max_pool_b = GlobalMaxPooling1D()(xc_b)

    x_a = concatenate([avg_pool_a, max_pool_a])
    x_a = BatchNormalization()(x_a)
    x_a = Dropout(0.3)(Dense(64, activation='relu')(x_a))

    x_b = concatenate([avg_pool_b, max_pool_b])
    x_b = BatchNormalization()(x_b)
    x_b = Dropout(0.1)(Dense(64, activation='relu')(x_b))

    x = concatenate([x_a, x_b])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(64, activation='relu')(x))

    x = Dense(2, activation="sigmoid")(x)
    model = Model(inputs=[inp_a, inp_b], outputs=x)

    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])

    model.fit([X_train_a, X_train_b], y_ohe,
              batch_size=64,
              epochs=20,
              validation_split=0.3,
              verbose=1,
              class_weight={0: 1, 1: 10},
              callbacks=[check_point, early_stop, tb_cb])
    model = load_model(model_path)
    return model


model = build_model(lr=1e-4, lr_d=0)
pred = model.predict([X_test_a, X_test_b], batch_size=1024)
predictions = np.round(np.argmax(pred, axis=1)).astype(int)

report = classification_report(y_test, predictions)

print(report)
