import json
import re

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
    """提取描述文本并jieba分词"""
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


def extract_all_kb_jieba():
    """提取描述文本并jieba分词"""
    with open(root_dir.format("kb_data.all.jieba.text.tsv"), "w", encoding="utf-8") as cutwt:
        kb_lines = open(root_dir.format("kb_data"), "r", encoding="utf-8").readlines()
        lall = 0
        for kb_line in tqdm(kb_lines):
            kbn = json.loads(kb_line)
            subject_id = kbn["subject_id"]

            all_str = ""
            all_str += "。".join(kbn["alias"])
            datas = kbn["data"]
            for data in datas:
                all_str += "。".join(data.values())
            all_str = all_str.replace("摘要", "。")

            all_str = re.sub(r'[\t\r\n]', '', all_str).strip()

            cut_all_str = " ".join(jieba.cut(all_str)).strip()
            lencut = len(cut_all_str.split(" "))
            cutwt.write(f"{subject_id}\t{all_str}\t{cut_all_str}\n")

            lall += lencut
        print("len entity all")
        print(lall/len(kb_lines))


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


def load_cut_text(file_type, col=1):
    """加载描述文本切词后的映射"""
    with open(root_dir.format(file_type), "r", encoding="utf-8") as cutrd:
        return {line.split("\t")[0]: line.split("\t")[col] for line in cutrd.readlines()}


def load_kb_predicate():
    """加载predicate列表"""
    pp = r"kb_data.predicate.tsv"
    with open(root_dir.format(pp), "r", encoding="utf-8") as predicaterd:
        return {line.split("\t")[0]: line.split("\t")[1] for line in predicaterd.readlines()}


def load_kb_tag(pp):
    """加载tag列表"""
    with open(root_dir.format(pp), "r", encoding="utf-8") as predicaterd:
        return {line.split("\t")[0]: line.split("\t")[2] for line in predicaterd.readlines()}


def extract_kb_predicate_lines():
    """提取关系文本"""
    with open(root_dir.format("kb_data.predicate.tsv"), "w", encoding="utf-8") as cutwt:
        lall = 0
        kb_lines = open(root_dir.format("kb_data"), "r", encoding="utf-8").readlines()
        for kb_line in tqdm(kb_lines):
            kbn = json.loads(kb_line)
            subject_id = kbn["subject_id"]
            datas = kbn["data"]
            predicate_text = [data["predicate"] for data in datas]
            lall = lall + len(predicate_text)
            text = " ".join(predicate_text)
            cutwt.write(f"{subject_id}\t{text.strip()}\n")
        print(lall/len(kb_lines))


def extract_kb_tag_lines():
    """
    提取标签文本
    """
    with open(root_dir.format("kb_data.tag.tsv"), "w", encoding="utf-8") as cutwt:
        kb_lines = open(root_dir.format("kb_data"), "r", encoding="utf-8").readlines()
        lall = 0
        for kb_line in tqdm(kb_lines):
            kbn = json.loads(kb_line)
            subject_id = kbn["subject_id"]
            tags = []
            datas = kbn["data"]

            for data in datas:
                if data["predicate"] == "标签":
                    tags.append(data["object"])
            text_line = " ".join(tags)
            cut_text_line = " ".join(jieba.cut(text_line))
            lall += len(cut_text_line.split(" "))
            cutwt.write(f"{subject_id}\t{text_line.strip()}\t{cut_text_line.strip()}\n")
        print(lall / len(kb_lines))


train_text_dic = load_cut_text("train.jieba.text.tsv")
test_text_dic = load_cut_text("train.jieba.text.tsv")
validate_text_dic = load_cut_text("validate.jieba.text.tsv")

all_text_dic = dict(dict(train_text_dic, **test_text_dic), **validate_text_dic)
kb_text_dic = load_cut_text("kb_data.jieba.text.tsv")
kb_predicate_dic = load_kb_predicate()

#  id:  tag
kb_tag_dic = load_kb_tag(r"kb_data.tag.tsv")
# 分词  id： text
kb_all_text_dic = load_cut_text("kb_data.all.jieba.text.tsv", col=2)
# 未分词  id： text
kb_all_text_dic_char = load_cut_text("kb_data.all.jieba.text.tsv", col=1)


def main():
    # extract_text_jieba("train")
    # extract_text_jieba("test")
    # extract_text_jieba("validate")
    # 12.481488888888888  query平均长度
    # extract_text_jieba("develop")
    # 55.98031068097342 描述平均长度
    # extract_kb_jieba()
    # len_count()
    # 平均边的数量   9.855715187400438
    # extract_kb_predicate_lines()
    # 平均标签数  1.8910287237133439
    # 平均标签词数  5.1393505855950625
    # extract_kb_tag_lines()
    # 实体平均总词数 104.49056986564877
    extract_all_kb_jieba()


if __name__ == '__main__':
    main()
