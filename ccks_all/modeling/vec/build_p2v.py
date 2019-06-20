from gensim.models import Word2Vec

vec_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\kb_data.predicate.vec.txt"

p_path = r"D:/data/biendata/ccks2019_el/ccks_train_data/kb_data.predicate.tsv"
sentences = open(p_path, "r", encoding="utf-8").readlines()
sentences = [[l.strip() for l in line.split(" ")[1:]] for line in sentences]

model = Word2Vec(min_count=1, size=300, workers=12, window=3, sg=1, hs=1)
model.build_vocab(sentences)

model.train(sentences, total_examples=model.corpus_count, epochs=20)

model.wv.save_word2vec_format(vec_path)
