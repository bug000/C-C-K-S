import csv

import pandas as pd

from matchzoo import Embedding


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


# embedding_path = "D:/data/word2vec/zh/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.utf8.txt"
# embedding_path = "D:/data/word2vec/zh/sgns.context.word-word.dynwin5.thr10.neg5.dim300.iter5/sgns.context.word-word.dynwin5.thr10.neg5.dim300.iter5.utf8.txt"
embedding_path = "D:/data/word2vec/zh/sgns.context.word-word.dynwin5.thr10.neg5.dim300.iter5/test.txt"
# glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
embedding = load_from_file(embedding_path)
print("embedding loaded")
