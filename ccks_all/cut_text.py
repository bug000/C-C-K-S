import json

import jieba
from tqdm import tqdm

jieba.initialize()

root_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\{}"


def extract_text_jieba(file_type):
    with open(root_dir.format(file_type + ".jieba.text.tsv"), "w", encoding="utf-8") as cutwt:
        json_lines = open(root_dir.format(file_type + ".json"), "r", encoding="utf-8").readlines()
        for json_line in json_lines:
            qdict = json.loads(json_line)
            text = qdict["text"]
            text = " ".join(jieba.cut(text))
            text_id = qdict["text_id"]
            cutwt.write(f"{text_id}\t{text.strip()}\n")


def extract_kb_jieba():
    with open(root_dir.format("kb_data.jieba.text.tsv"), "w", encoding="utf-8") as cutwt:
        kb_lines = open(root_dir.format("kb_data"), "r", encoding="utf-8").readlines()
        for kb_line in tqdm(kb_lines):
            kbn = json.loads(kb_line)
            subject_id = kbn["subject_id"]
            text = ""
            datas = kbn["data"]
            for data in datas:
                if data["predicate"] == "摘要" or data["predicate"] == "义项描述":
                    text += data["object"]
            text = " ".join(jieba.cut(text))
            cutwt.write(f"{subject_id}\t{text.strip()}\n")


def len_count():
    with open(root_dir.format("train.jieba.text.tsv"), "r", encoding="utf-8") as cr:
        # with open(root_dir.format("kb_data.jieba.text.tsv"), "r", encoding="utf-8") as cr:
        cr = cr.readlines()
        tal = 0
        for line in cr:
            tnub = len(line.split("\t")[1].split(" "))
            tal += tnub

        mean = tal / len(cr)
        print(mean)


def load_cut_text(file_type):
    with open(root_dir.format(file_type), "r", encoding="utf-8") as cutrd:
        return {line.split("\t")[0]: line.split("\t")[1] for line in cutrd.readlines()}


train_text_dic = load_cut_text("train.jieba.text.tsv")
test_text_dic = load_cut_text("train.jieba.text.tsv")
validate_text_dic = load_cut_text("validate.jieba.text.tsv")

all_text_dic = dict(dict(train_text_dic, **test_text_dic), **validate_text_dic)
kb_text_dic = load_cut_text("kb_data.jieba.text.tsv")


def main():
    # extract_text_jieba("train")
    # extract_text_jieba("test")
    # extract_text_jieba("validate")
    # 12.481488888888888
    # extract_text_jieba("develop")
    # 55.98031068097342
    # extract_kb_jieba()
    len_count()


if __name__ == '__main__':
    main()
