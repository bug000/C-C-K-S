import json

train_file_path = r"Z:/research/data/biendata/ccks2019_el/ccks2019_el/train.json"

jll = 0
with open(train_file_path, 'r', encoding='utf-8') as all_trains:
    for json_line in all_trains:
        corpu_data = json.loads(json_line)
        mention_data = corpu_data["mention_data"]
        text = corpu_data["text"].replace("[\n\t\r]", "")
        jl = len(text)
        jll += jl

print(jll/len(open(train_file_path, 'r', encoding='utf-8').readlines()))
