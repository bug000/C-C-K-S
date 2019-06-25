from kashgari.tasks.seq_labeling import BLSTMCRFModel
from kashgari.embeddings import BERTEmbedding
from keras.callbacks import EarlyStopping, TensorBoard

from ccks_ner import dload

train_x, train_y = dload.load_json_data('train')
validate_x, validate_y = dload.load_json_data('validate')
test_x, test_y = dload.load_json_data('test')

print(f"train data count: {len(train_x)}")
print(f"validate data count: {len(validate_x)}")
print(f"test data count: {len(test_x)}")

model_path_o = r"D:\data\biendata\ccks2019_el\ner\m0"
model_path_n = r"D:\data\biendata\ccks2019_el\ner\m0.0"
log_filepath = r"D:\data\biendata\ccks2019_el\ner\m0.0log"

# emn_path = r'D:\data\bert\chinese_L-12_H-768_A-12'
emn_path = r'D:\data\bert\chinese-bert_chinese_wwm_L-12_H-768_A-12'

# check_point = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=1)
# early_stop = EarlyStopping(monitor="val_crf_accuracy", mode="max", patience=2)

log = TensorBoard(log_dir=log_filepath, write_images=False, write_graph=True, histogram_freq=0)


model = BLSTMCRFModel.load_model(model_path_o)


model.fit(train_x,
          train_y,
          x_validate=validate_x,
          y_validate=validate_y,
          epochs=20,
          batch_size=512,
          labels_weight=True,
          fit_kwargs={'callbacks': [early_stop, log]})

model.evaluate(test_x, test_y)

model.save(model_path_n)


"""
继续训练
"""

