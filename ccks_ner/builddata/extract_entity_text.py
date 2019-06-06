import json

from tqdm import tqdm


def extract_entity_text(entity_json_line: dict) -> str:
    """
    得到 entity 描述文本
    :param entity_json_line:
    :return:
    """
    all_str = ""
    all_str += "。".join(entity_json_line["alias"])
    datas = entity_json_line["data"]
    for data in datas:
        all_str += "。".join(data.values())
    all_str = all_str.replace("摘要", "。")
    return all_str


entity_id_path = r"D:/data/biendata/ccks2019_el/ccks_train_data/clf_data/all.tsv"

entity_id_text_path = r"D:/data/biendata/ccks2019_el/ccks_train_data/clf_data/all.text.tsv"

all_train_path = r"D:/data/biendata/ccks2019_el/ccks2019_el/train.json"
all_kg_path = r"D:/data/biendata/ccks2019_el/ccks_train_data/kb_data"

entity_text_dict = {
    json.loads(entity_line)["subject_id"]: extract_entity_text(json.loads(entity_line)).replace("[\t\n\r]", "", 100)
    for entity_line in open(all_kg_path, "r", encoding="utf-8").readlines()
}

query_text_dict = {
    json.loads(query_line)["text_id"]: json.loads(query_line)["text"].replace("[\t\n\r]", "", 100)
    for query_line in open(all_train_path, "r", encoding="utf-8").readlines()
}

with open(entity_id_text_path, "w", encoding="utf-8") as text_wt:
    # for id_line in set(open(entity_id_path, "r", encoding="utf-8").readlines()):
    for id_line in tqdm(open(entity_id_path, "r", encoding="utf-8").readlines()):
        label, text_id, entity_id = id_line.strip().split("\t")
        text_wt.write(f"{label}\t{text_id}\t{entity_id}\t{query_text_dict[text_id]}\t{entity_text_dict[entity_id]}\n")
        text_wt.flush()
    text_wt.close()
