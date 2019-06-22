import csv
import json

import dill
import pandas as pd

import matchzoo as mz
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from matchzoo import Embedding
from matchzoo.models import ConvKNRM
from tqdm import tqdm

from ccks_all.cut_text import all_text_dic, kb_all_text_dic
from ccks_all.modeling.pair.pair_model import Metrics

root_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\{}"


def load_data(data_type, line_nub=-1):

    id_set = set()

    X_left = []
    X_left_id = []
    X_right = []
    X_right_id = []

    y = []

    json_line_s = open(root_dir.format(data_type + ".json.jieba.pre.json"), "r", encoding="utf-8").readlines()
    query_data_loder = tqdm(json_line_s)
    query_data_loder.set_description("load query data lines")
    for json_line in query_data_loder:
        tdata = json.loads(json_line)
        text_id = tdata["text_id"]
        query_text = all_text_dic[text_id]
        mention_data = tdata["mention_data"]
        for mention in mention_data:

            kb_id_subject = text_id + mention["mention"]
            kb_id = mention["kb_id"]
            doc_text = kb_all_text_dic[kb_id]

            y_label = int(mention["label"])

            pid = text_id + "_" + kb_id

            if len(id_set) == line_nub:
                break

            if pid not in id_set:
                id_set.add(pid)

                X_left.append(query_text)
                X_right.append(doc_text)
                X_left_id.append(text_id)
                X_right_id.append(kb_id_subject)
                y.append(y_label)

        else:
            continue
        break

    df = pd.DataFrame({
        'text_left': X_left,
        'text_right': X_right,
        'id_left': X_left_id,
        'id_right': X_right_id,
        'label': y
    })
    return mz.pack(df)


def load_from_file(file_path: str, mode: str = 'word2vec') -> Embedding:
    """
    Load embedding from `file_path`.

    :param file_path: Path to file.
    :param mode: Embedding file format mode, one of 'word2vec' or 'glove'.
        (default: 'word2vec')
    :return: An :class:`matchzoo.embedding.Embedding` instance.
    """
    if mode == 'word2vec':
        data = pd.read_csv(file_path,
                           sep=" ",
                           index_col=0,
                           header=None,
                           # na_filter=False,
                           skipinitialspace=True,
                           # delim_whitespace=True,
                           skiprows=1)
        # data = data.str.strip()
        data = data.dropna(axis=1)
    elif mode == 'glove':
        data = pd.read_csv(file_path,
                           sep=" ",
                           index_col=0,
                           header=None,
                           quoting=csv.QUOTE_NONE)
    else:
        raise TypeError(f"{mode} is not a supported embedding type."
                        f"`word2vec` or `glove` expected.")
    return Embedding(data)


def append_params_to_readme(model):
    import tabulate

    with open('README.rst', 'a+') as f:
        subtitle = model.params['model_class'].__name__
        line = '#' * len(subtitle)
        subtitle = subtitle + '\n' + line + '\n\n'
        f.write(subtitle)

        df = model.params.to_frame()[['Name', 'Value']]
        table = tabulate.tabulate(df, tablefmt='rst', headers='keys') + '\n\n'
        f.write(table)


print('data loading ...')
train_pack_raw = load_data('train', -1)
dev_pack_raw = load_data('validate', 100000)
test_pack_raw = load_data('test', 100000)


ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=1),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print("`ranking_task` initialized with metrics", ranking_task.metrics)

print("loading embedding ...")
# embedding_path = "D:/data/word2vec/zh/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.utf8.txt"
embedding_path = "D:/data/word2vec/zh/sgns.context.word-word.dynwin5.thr10.neg5.dim300.iter5/sgns.context.word-word.dynwin5.thr10.neg5.dim300.iter5.utf8.txt"

# glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
# embedding = mz.embedding.load_from_file(embedding_path)
embedding = load_from_file(embedding_path)
print("embedding loaded")

preprocessor = mz.preprocessors.BasicPreprocessor(
    fixed_length_left=20,
    fixed_length_right=200,
    remove_stop_words=False
)

train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)


# model = mz.models.MVLSTM()
model = ConvKNRM()
# model = mz.contrib.models.MatchLSTM()
model.params.update(preprocessor.context)
model.params['task'] = ranking_task
model.params['embedding_output_dim'] = embedding.output_dim
model.params['embedding_trainable'] = False
model.params['filters'] = 256
model.params['conv_activation_func'] = 'tanh'
model.params['max_ngram'] = 3
model.params['use_crossmatch'] = True
model.params['kernel_num'] = 21
model.params['sigma'] = 0.1
model.params['exact_sigma'] = 0.001
model.params['optimizer'] = 'adadelta'
model.build()
model.compile()

model.backend.summary()

embedding_matrix = embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
model.load_embedding_matrix(embedding_matrix)

test_x, test_y = test_pack_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=test_x, y=test_y, batch_size=len(test_x))

train_generator = mz.DataGenerator(
    train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=1,
    batch_size=8
)
print('num batches:', len(train_generator))

model_path = r"D:/data/biendata/ccks2019_el/entityrank/m1/model/"
early_stop = EarlyStopping(monitor=mz.metrics.NormalizedDiscountedCumulativeGain(k=1), mode="max", patience=3)
check_point = ModelCheckpoint(model_path + "best_model.h5",
                              monitor=mz.metrics.NormalizedDiscountedCumulativeGain(k=1),
                              verbose=1, save_best_only=True, mode="min")
history = model.fit_generator(train_generator, epochs=100, callbacks=[evaluate, early_stop, check_point])


model.save(model_path)
dill.dump(preprocessor,  open(model_path + "preprocessor.dill", mode='wb'))

append_params_to_readme(model)


