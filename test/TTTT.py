import keras
import pandas as pd
import numpy as np
import matchzoo as mz
import json

print('matchzoo version', mz.__version__)
print()

print('data loading ...')
train_pack_raw = mz.datasets.wiki_qa.load_data('train', task='ranking')
dev_pack_raw = mz.datasets.wiki_qa.load_data('dev', task='ranking', filtered=True)
test_pack_raw = mz.datasets.wiki_qa.load_data('test', task='ranking', filtered=True)
print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')

ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print("`ranking_task` initialized with metrics", ranking_task.metrics)

print("loading embedding ...")
glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
print("embedding loaded as `glove_embedding`")


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


preprocessor = mz.preprocessors.BasicPreprocessor(
    fixed_length_left=10,
    fixed_length_right=40,
    remove_stop_words=False
)

train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)

model = mz.contrib.models.MatchLSTM()
model.params.update(preprocessor.context)
model.params['task'] = ranking_task
model.params['embedding_output_dim'] = 100
model.params['embedding_trainable'] = False
model.params['fc_num_units'] = 100
model.params['lstm_num_units'] = 100
model.params['dropout_rate'] = 0.5
model.params['optimizer'] = 'adadelta'
model.guess_and_fill_missing_params()
model.build()
model.compile()

model.backend.summary()

embedding_matrix = glove_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
model.load_embedding_matrix(embedding_matrix)

test_x, test_y = test_pack_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=test_x, y=test_y, batch_size=len(test_x))

train_generator = mz.DataGenerator(
    train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=1,
    batch_size=20
)
print('num batches:', len(train_generator))

history = model.fit_generator(train_generator, epochs=10, callbacks=[evaluate], workers=4, use_multiprocessing=False)

append_params_to_readme(model)
