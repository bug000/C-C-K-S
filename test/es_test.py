import re

from ccks_all.esret import jaccard_sim, expand_sim_es


def dsc_token(text: str):
    fds = re.findall(r"《([^《|》]*)》", text)
    for fd_str in fds:
        fd_str = fd_str.strip()
        ind = text.find(fd_str)
        yield fd_str, ind


# que = "《会计制度设计(第五版)》(李端生 编)【简介_书评"
que = "《枫之谷2》可爱冒险登场!今年内韩服开测"
que = "《枫之谷2》可爱冒险登场!今年内韩服开测"
dsc_entity = dsc_token(que)

for keyword in dsc_entity:
    sres = expand_sim_es(keywords=keyword[0], index_str="subject_text", jaccard_filt=0.6)
    for sre in sres:
        print(str(sre))
        print(jaccard_sim(keyword[0], sre))







