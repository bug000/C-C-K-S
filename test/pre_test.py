import json
import re

from ccks_all.esret import jaccard_sim, expand_sim_es
from ccks_all.predict.predict import BiLSTMCRFPredicter


def dsc_token(text: str):
    fds = re.findall(r"《([^《|》]*)》", text)
    for fd_str in fds:
        fd_str = fd_str.strip()
        ind = text.find(fd_str)
        yield fd_str, ind


# que = "《会计制度设计(第五版)》(李端生 编)【简介_书评"
que = "《枫之谷2》可爱冒险登场!今年内韩服开测"
dsc_entity = dsc_token(que)

crf_model_path = r"D:\data\biendata\ccks2019_el\ner\m30not"
crfer = BiLSTMCRFPredicter(crf_model_path, type_filter=False, save_label=True, batch=1024)

json_lines = ["""
{"text_id": "18380", "text": "民国大军阀 最新章节 无弹窗广告", "mention_data": [{"kb_id": "291585", "mention": "民国大军阀", "offset": "0"}, {"kb_id": "397849", "mention": "章节", "offset": "8"}, {"kb_id": "254121", "mention": "无", "offset": "11"}, {"kb_id": "381830", "mention": "弹窗", "offset": "12"}, {"kb_id": "69522", "mention": "广告", "offset": "14"}]}

"""]
json_lines = [json.loads(j) for j in json_lines]
pre_lines = crfer.predict(json_lines)

print(pre_lines)
pass



