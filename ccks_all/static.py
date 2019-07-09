import json
from typing import List

from elasticsearch import Elasticsearch

from tqdm import tqdm

es_servers = [{
    "host": "localhost",
    "port": "9200"
}]
es = Elasticsearch(es_servers)


def load_entiy():
    subject_dict = {}
    subject_id_dict = {}
    id2entity = {}
    kb_path = "D:/data/biendata/ccks2019_el/ccks_train_data/kb_data"
    kr_reader = tqdm(open(kb_path, 'r', encoding='utf-8').readlines())
    kr_reader.set_description("load kb data")
    for line in kr_reader:
        # 小写
        knobj = json.loads(line.lower())

        id2entity[knobj["subject_id"]] = knobj

        """subject"""
        subject = knobj["subject"]
        subject_id = knobj["subject_id"]
        if subject in subject_id_dict.keys():
            if subject_id not in subject_id_dict[subject]:
                subject_id_dict[subject].append(subject_id)
                subject_dict[subject].append(knobj)
        else:
            subject_id_dict[subject] = [subject_id]
            subject_dict[subject] = [knobj]

        """alias"""
        for sub_alias in knobj["alias"]:
            if sub_alias in subject_id_dict.keys():
                if subject_id not in subject_id_dict[sub_alias]:
                    subject_id_dict[sub_alias].append(subject_id)
                    subject_dict[sub_alias].append(knobj)
            else:
                subject_id_dict[sub_alias] = [subject_id]
                subject_dict[sub_alias] = [knobj]
    return id2entity, subject_dict, subject_id_dict


"""
id:obj
text:obj
"""
id2entity, subject_dict, subject_id_dict = load_entiy()


class SubjectIndex(object):

    def __init__(self):
        self.subject_index_dict = {}
        self.id2entity = id2entity
        for id in self.id2entity.keys():
            knobj = id2entity[id]
            subject = knobj["subject"]
            self.add(subject, id)
            for sub_alias in knobj["alias"]:
                self.add(sub_alias, id)

    def add(self, subject_text, subject_id):
        for c in subject_text:
            if c in self.subject_index_dict.keys():
                self.subject_index_dict[c].append(subject_id)
            else:
                self.subject_index_dict[c] = [subject_id]

    def retrivl(self, mention_text):
        sid_count_dic = {}
        for c in mention_text:
            sids = self.subject_index_dict[c]
            for sid in sids:
                if sid in sid_count_dic.keys():
                    sid_count_dic[sid] += 1
                else:
                    sid_count_dic[sid] = 0
        sorted_sid_count_dic = sorted(sid_count_dic.items(), key=lambda item: item[1], reverse=True)
        return sorted_sid_count_dic[0]

# subject_index = SubjectIndex()
def build_es_index():
    index = "subject_text"
    line_iter = tqdm(subject_dict.keys())
    line_iter.set_description("subject index")
    for subject in line_iter:
        doc_body = {
            "enti": subject
        }
        es.index(index=index, doc_type="gl", body=doc_body)
