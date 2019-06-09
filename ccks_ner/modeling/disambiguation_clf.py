from kashgari import macros
from kashgari.tasks.classification import BLSTMModel
from kashgari.tasks.classification import DropoutBGRUModel
from kashgari.embeddings import BERTEmbedding, WordEmbeddings
from kashgari.tasks.seq_labeling import CNNLSTMModel
from keras.callbacks import EarlyStopping, TensorBoard

from ccks_ner import dload


class _Config(object):
    def __init__(self):
        self.use_CuDNN_cell = True
        self.sequence_labeling_tokenize_add_bos_eos = False


macros.config = _Config()


train_x, train_y = dload.load_entity_clf_cut_data('test')
validate_x, validate_y = dload.load_entity_clf_cut_data('validate')
test_x, test_y = dload.load_entity_clf_cut_data('test')

print(f"train data count: {len(train_x)}")
print(f"validate data count: {len(validate_x)}")
print(f"test data count: {len(test_x)}")

model_path = r"D:\data\biendata\ccks2019_el\clf_weight_model"
log_filepath = r"D:\data\biendata\ccks2019_el\clf_log"

early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2)
# early_stop = EarlyStopping(monitor="val_acc", mode="max", patience=2)

log = TensorBoard(log_dir=log_filepath, write_images=False, write_graph=True, histogram_freq=0)

emn_path = r'D:\data\bert\chinese_L-12_H-768_A-12'
embedding = BERTEmbedding(emn_path, sequence_length=1024)
emn_path = r'D:/data/word2vec/zh/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.utf8.txt'
embedding = WordEmbeddings(emn_path, sequence_length=1024)

# model = DropoutBGRUModel(embedding)
model = BLSTMModel(embedding)


model.fit(train_x[:100000],
          train_y[:100000],
          x_validate=validate_x[:20000],
          y_validate=validate_y[:20000],
          epochs=20,
          batch_size=256,
          labels_weight=True,
          fit_kwargs={'callbacks': [early_stop, log]})

model.evaluate(test_x, test_y)

model.save(model_path)


