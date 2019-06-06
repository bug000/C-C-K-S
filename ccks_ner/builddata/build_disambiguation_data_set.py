#!/usr/bin/python
# -*- coding: UTF-8 -*-

import json


def get_kb_sub_dic():
    subject_dict = {}
    kb_path = "D:/data/biendata/ccks2019_el/ccks_train_data/kb_data"
    kr_reader = open(kb_path, 'r', encoding='utf-8')
    for line in kr_reader:
        # 小写
        knobj = json.loads(line.lower())

        """subject"""
        subject = knobj["subject"]
        if subject in subject_dict.keys():
            subject_dict[subject].append(knobj)
        else:
            subject_dict[subject] = [knobj]

        """alias"""
        for alias_sub in knobj["alias"]:
            if alias_sub in subject_dict.keys():
                subject_dict[alias_sub].append(knobj)
            else:
                subject_dict[alias_sub] = [knobj]

    return subject_dict


def get_kb_dic():
    subject_dict = {}
    kb_path = "D:/data/biendata/ccks2019_el/ccks_train_data/kb_data"
    kr_reader = open(kb_path, 'r', encoding='utf-8')
    for line in kr_reader:
        # 小写
        knobj = json.loads(line.lower())
        for sub_alias in knobj["alias"]:
            if sub_alias in subject_dict:
                subject_dict[sub_alias].append(knobj)
            else:
                subject_dict[sub_alias] = [knobj]
    return subject_dict


def main():

    c_set = set()

    clf_data_dir_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\clf_data\{}"

    clf_data_writer = open(clf_data_dir_path.format("all.tsv"), 'w', encoding="utf-8")

    # subject_dict = get_kb_dic()
    subject_dict = get_kb_sub_dic()

    json_path = "D:/data/biendata/ccks2019_el/ccks2019_el/train.json"
    json_line_s = open(json_path, "r", encoding="utf-8").readlines()
    for json_line in json_line_s:
        entity_dict = json.loads(json_line.lower())
        # 得到实体 id
        mention_data_s = entity_dict["mention_data"]
        text_id = entity_dict["text_id"]
        for mention_data in mention_data_s:
            pos_subject_id = mention_data["kb_id"]
            if pos_subject_id != "nil":
                mention_text = mention_data["mention"]
                if mention_text in subject_dict:
                    entirs = subject_dict[mention_text]
                    for entity_dict in entirs:
                        subject_id = entity_dict["subject_id"]
                        if subject_id == pos_subject_id:
                            id_line = f"{1}\t{text_id}\t{subject_id}\n"
                            if id_line not in c_set:
                                c_set.add(id_line)
                                clf_data_writer.write(id_line)
                        else:
                            id_line = f"{0}\t{text_id}\t{subject_id}\n"
                            if id_line not in c_set:
                                c_set.add(id_line)
                                clf_data_writer.write(id_line)
                        clf_data_writer.flush()
                else:
                    print(mention_text)

    clf_data_writer.close()


if __name__ == '__main__':
    main()

