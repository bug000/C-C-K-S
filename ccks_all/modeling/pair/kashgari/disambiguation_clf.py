import random
import numpy as np

from itertools import chain

from kashgari import macros
from kashgari.tasks.classification import BLSTMModel
from kashgari.tasks.classification import DropoutBGRUModel
from kashgari.embeddings import BERTEmbedding, WordEmbeddings
from kashgari.tasks.seq_labeling import CNNLSTMModel
from keras.callbacks import EarlyStopping, TensorBoard
from keras.preprocessing import sequence

from ccks_all.modeling import datas


class _Config(object):
    def __init__(self):
        self.use_CuDNN_cell = True
        self.sequence_labeling_tokenize_add_bos_eos = False


macros.config = _Config()


train_x, train_y = datas.load_pair_clf_data('train.json.jieba.default.pre.json', 200_0000)
validate_x, validate_y = datas.load_pair_clf_data('validate.json.jieba.default.pre.json')
# test_x, test_y = datas.load_pair_clf_data('test.json.jieba.search.pre.json')


print(f"train data count: {len(train_x)}")
print(f"validate data count: {len(validate_x)}")
# print(f"test data count: {len(test_x)}")

model_path = r"D:\data\biendata\ccks2019_el\kashgari\m0"
log_filepath = r"D:\data\biendata\ccks2019_el\kashgari\m0log"

early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2)
# early_stop = EarlyStopping(monitor="val_acc", mode="max", patience=2)

log = TensorBoard(log_dir=log_filepath, write_images=False, write_graph=True, histogram_freq=0)

# emn_path = r'D:\data\bert\chinese_L-12_H-768_A-12'
emn_path = r'D:\data\bert\chinese-bert_chinese_wwm_L-12_H-768_A-12'
embedding = BERTEmbedding(emn_path, sequence_length=512)


# model = DropoutBGRUModel(embedding)
model = BLSTMModel(embedding)
model.fit(train_x,
          train_y,
          x_validate=validate_x,
          y_validate=validate_y,
          epochs=20,
          batch_size=128,
          labels_weight=True,
          fit_kwargs={'callbacks': [early_stop, log]})

model.save(model_path)


test_x, test_y = datas.load_pair_clf_data('test.json.jieba.search.pre.json')
model.evaluate(test_x, test_y)




