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


def load_pair_clf_data_generator(kashgari_model, data_nub,
                       data_type: str,
                       batch_size: int = 64,
                       is_bert: bool = False,
                       line_nub=-1):

    while True:
        page_list = list(range((data_nub // batch_size) + 1))
        random.shuffle(page_list)
        for page in page_list:
            start_index = page * batch_size
            end_index = start_index + batch_size
            target_x = []
            for x in x_data:
                target_x.append(x[start_index: end_index])
            target_y = y_data[start_index: end_index]
            if len(target_x[0]) == 0:
                target_x.pop()
                for x in x_data:
                    target_x.append(x[0: batch_size])
                target_y = y_data[0: batch_size]

            padded_x = []
            for i, x in enumerate(target_x):
                tokenized_x = kashgari_model.embedding.tokenize(x)

                if isinstance(kashgari_model.embedding.sequence_length, int):
                    padded_x.append(sequence.pad_sequences(tokenized_x,
                                                           maxlen=kashgari_model.embedding.sequence_length,
                                                           padding='post')
                                    )
                elif isinstance(kashgari_model.embedding.sequence_length, list):
                    padded_x.append(sequence.pad_sequences(tokenized_x,
                                                           maxlen=kashgari_model.embedding.sequence_length[i],
                                                           padding='post')
                                    )

            if kashgari_model.multi_label:
                padded_y = kashgari_model.multi_label_binarizer.fit_transform(target_y)
            else:
                tokenized_y = kashgari_model.convert_label_to_idx(target_y)
                padded_y = kashgari_model.to_categorical(tokenized_y,
                                          num_classes=len(kashgari_model.label2idx),
                                          dtype=np.int)
            if is_bert:
                if isinstance(kashgari_model.embedding.sequence_length, int):
                    padded_x_seg = [np.zeros(shape=(len(padded_x_i),
                                                    kashgari_model.embedding.sequence_length))
                                    for padded_x_i in padded_x]
                elif isinstance(kashgari_model.embedding.sequence_length, list):
                    padded_x_seg = [np.zeros(shape=(len(padded_x_i),
                                                    kashgari_model.embedding.sequence_length[i]))
                                    for i, padded_x_i in enumerate(padded_x)]
                x_input_data = list(chain(*[(x, x_seg)
                                            for x, x_seg in zip(padded_x, padded_x_seg)]))
            else:
                x_input_data = padded_x[0] if x_data_level == 2 else padded_x
            yield (x_input_data, padded_y)


train_x, train_y = datas.load_pair_clf_data('train.json.jieba.search.pre.json', 1000000)
validate_x, validate_y = datas.load_pair_clf_data('validate.json.jieba.search.pre.json')
test_x, test_y = datas.load_pair_clf_data('test.json.jieba.search.pre.json')



print(f"train data count: {len(train_x)}")
print(f"validate data count: {len(validate_x)}")
print(f"test data count: {len(test_x)}")

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
model.build_model(x_train, y_train, x_validate, y_validate)

model.model.fit_generator()
model.fit(train_x,
          train_y,
          x_validate=validate_x,
          y_validate=validate_y,
          epochs=20,
          batch_size=128,
          labels_weight=True,
          fit_kwargs={'callbacks': [early_stop, log]})

model.evaluate(test_x, test_y)

model.save(model_path)


