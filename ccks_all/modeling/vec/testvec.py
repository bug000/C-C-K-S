from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import datapath

vec_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\kb_data.predicate.vec.txt"

# model = Word2Vec()
model = KeyedVectors.load_word2vec_format(datapath(vec_path), binary=False)


# sim = model.similarity["外文名", "性别"]
sim = model.most_similar(positive=["烂番茄新鲜度"])
print(sim)
