import pickle

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, SpatialDropout2D
from keras.models import Model, load_model
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report


embedding_path = "D:/data/word2vec/zh/test.txt"

train_dir = r"D:\data\biendata\ccks2019_el\entityclf\m4\{}"
# train_dir = r"C:\python3workspace\kera_ner_demo\ccks_ner\modeling\pair_model\dt\m3\{}"
log_filepath = train_dir.format(r"log")
toka_path = train_dir.format(r"\toka.bin")
model_path = train_dir.format(r"bilstm_model.hdf5")

root_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\clf_data\split\{}"
# train = pd.read_csv(root_path.format("train.tsv.hanlp.tsv"), sep="\t", nrows=20000, header=None)
test = pd.read_csv(root_path.format("test.tsv.hanlp.tsv"), sep="\t", header=None)
# val = pd.read_csv(root_path.format("validate.tsv.hanlp.tsv"), sep="\t", nrows=100, header=None)


y_test = test.iloc[:, 0]


tk = pickle.load(open(toka_path, 'rb'))

test_a_tokenized = tk.texts_to_sequences(test.iloc[:, 3].values)
test_b_tokenized = tk.texts_to_sequences(test.iloc[:, 4].values)

max_len_a = 50
max_len_b = 800
X_test_a = pad_sequences(test_a_tokenized, maxlen=max_len_a)
X_test_b = pad_sequences(test_b_tokenized, maxlen=max_len_b)

model = load_model(model_path)
pred = model.predict([X_test_a, X_test_b], batch_size=512)
predictions = np.round(np.argmax(pred, axis=1)).astype(int)

report = classification_report(y_test, predictions)

print(report)
