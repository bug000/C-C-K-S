from kashgari.embeddings import BERTEmbedding

emn_path = r'D:\data\bert\chinese_L-12_H-768_A-12'
embedding = BERTEmbedding(emn_path, sequence_length=512)


def trans_clf_data(data_type: str):
    data_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\clf_data\split\{}.tsv"
    bert_data_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\clf_data\split\{}.tsv.bert.tsv"
    with open(data_dir.format(data_type), 'r', encoding='utf-8') as data_rd:
        with open(bert_data_dir.format(data_type), 'w', encoding='utf-8') as data_wt:
            for line in data_rd:
                emb = embedding.embed(line)
                pass


trans_clf_data("train")
