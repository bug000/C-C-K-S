import json
import jieba
from pyhanlp import *
from tqdm import tqdm

BEGIN = "B-{}"
MIDD = "M-{}"
END = "E-{}"
SINGLE = "S-{}"
NONE = "O"


def get_kb_type_dic():
    subject_dict = {}
    kb_path = "D:/data/biendata/ccks2019_el/ccks_train_data/kb_data"
    kr_reader = open(kb_path, 'r', encoding='utf-8')
    for line in kr_reader:
        knobj = json.loads(line)
        for sub_alias in knobj["alias"]:
            subject_dict[sub_alias] = knobj["type"]
    return subject_dict


kb_type_dict = get_kb_type_dic()


def load_data(data_type):
    data_dir = r"C:\python3workspace\kera_ner_demo\ccks_ner\ccks_train_data\{}.tsv"
    fr = open(data_dir.format(data_type), 'r', encoding='utf-8')
    data_x = []
    data_y = []
    for line in fr:

        xl, yl = line.rstrip('\n').split("\t")
        if len(list(xl)) == len(list(yl)):
            data_x.append(list(xl))
            data_y.append(list(yl))
        else:
            print(line)
    return data_x, data_y


def load_json_data(data_type):
    data_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\{}.json"
    fr = open(data_dir.format(data_type), 'r', encoding='utf-8')
    data_x = []
    data_y = []
    for json_line in fr:
        corpu_data = json.loads(json_line)
        mention_data = corpu_data["mention_data"]
        text = corpu_data["text"]

        x_line = list(text)
        y_line = [NONE for _ in text]

        for men in mention_data:
            subject_types = kb_type_dict.get(men["mention"], [])
            men_len = len(men["mention"])
            men_offset = int(men["offset"])
            for type_str in subject_types:
                # 判断长度/如果长度为1则标注 single
                if men_len == 1:
                    y_line[men_offset] = SINGLE.format(type_str)
                else:
                    for i in range(men_offset, men_offset + men_len):
                        y_line[i] = MIDD.format(type_str)
                    y_line[men_offset + men_len - 1] = END.format(type_str)
                    y_line[men_offset] = BEGIN.format(type_str)

        data_x.append(x_line)
        data_y.append(y_line)
    return data_x, data_y


def load_entity_clf_data(data_type: str):
    data_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\clf_data\split\{}.tsv"
    data_x = []
    data_y = []
    with open(data_dir.format(data_type), 'r', encoding='utf-8') as data_rd:
        for i, line in enumerate(data_rd):
            ls = line.split("\t")
            label = ls[0].strip()
            text_a = ls[3].strip()
            text_b = ls[4].strip()
            x_line = []
            x_line.extend(list(text_a))
            x_line.append("CLS")
            x_line.extend(list(text_b))

            data_x.append(x_line)
            data_y.append(label)

    return data_x, data_y


def load_entity_clf_cut_data(data_type: str):
    data_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\clf_data\split\{}.tsv.hanlp.tsv"
    data_x = []
    data_y = []
    with open(data_dir.format(data_type), 'r', encoding='utf-8') as data_rd:
        for i, line in enumerate(tqdm(data_rd.readlines())):
            ls = line.split("\t")
            label = ls[0].strip()
            text_a = ls[3].strip()
            text_b = ls[4].strip()
            x_line = []
            # x_line.extend([term.word for term in HanLP.segment(text_a)])
            x_line.extend(text_a.split(" "))
            x_line.append("CLS")
            # x_line.extend([term.word for term in HanLP.segment(text_b)])
            x_line.extend(text_b.split(" "))
            data_x.append(x_line)
            data_y.append(label)

    return data_x, data_y
