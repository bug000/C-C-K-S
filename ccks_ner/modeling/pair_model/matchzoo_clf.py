import matchzoo as mz


preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=10, fixed_length_right=40, remove_stop_words=False)
train_pack_processed = preprocessor.fit_transform(train_pack_raw)
valid_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)

ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))

model = mz.models.KNRM()
model.params.update(preprocessor.context)
model.params['task'] = ranking_task
model.params['embedding_output_dim'] = glove_embedding.output_dim
model.params['embedding_trainable'] = True
model.params['kernel_num'] = 21
model.params['sigma'] = 0.1
model.params['exact_sigma'] = 0.001
model.params['optimizer'] = 'adadelta'
model.build()
model.compile()
#model.backend.summary()

history = model.fit_generator(train_generator, epochs=20, callbacks=[evaluate], workers=5, use_multiprocessing=False)





