import json
from tqdm import tqdm

from ccks_all.static import id2entity


def load_json_data(data_type):

    BEGIN = "B-{}"
    MIDD = "M-{}"
    END = "E-{}"
    SINGLE = "S-{}"
    NONE = "O"

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
            kb_id = men["kb_id"]
            if kb_id != "NIL":
                subject_types = id2entity.get(kb_id).get("type", [])
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


def load_json_data_no_type(data_type):
    BEGIN = "B"
    ENTITY = "I"
    NONE = "O"

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
            kb_id = men["kb_id"]
            if kb_id != "NIL":
                men_len = len(men["mention"])
                men_offset = int(men["offset"])

                for i in range(men_offset, men_offset + men_len):
                    y_line[i] = ENTITY
                y_line[men_offset + men_len - 1] = ENTITY
                y_line[men_offset] = BEGIN

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
