import matchzoo as mz
print(mz.__version__)


task = mz.tasks.Ranking()
print(task)


train_raw = mz.datasets.toy.load_data(stage='train', task=task)
test_raw = mz.datasets.toy.load_data(stage='test', task=task)

print(type(train_raw))

