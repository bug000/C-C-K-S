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

model_path = r"D:\data\biendata\ccks2019_el\ner_weight_model"
log_filepath = r"D:\data\biendata\ccks2019_el\ner_weight_log"

emn_path = r'D:\data\bert\chinese_L-12_H-768_A-12'

# check_point = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

# early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2)
early_stop = EarlyStopping(monitor="val_crf_accuracy", mode="max", patience=2)

log = TensorBoard(log_dir=log_filepath, write_images=False, write_graph=True, histogram_freq=0)

embedding = BERTEmbedding(emn_path, 50)

# 还可以选择 `BLSTMModel` 和 `CNNLSTMModel`
model = BLSTMCRFModel(embedding)

model.__base_hyper_parameters__ = {
        'lstm_layer': {
            'units': 256,
            'return_sequences': True
        },
        'dense_layer': {
            'units': 64,
            'activation': 'tanh'
        }
    }

model.fit(train_x,
          train_y,
          x_validate=validate_x,
          y_validate=validate_y,
          epochs=20,
          batch_size=1024,
          labels_weight=True,
          fit_kwargs={'callbacks': [early_stop, log]})

model.evaluate(test_x, test_y)

model.save(model_path)
