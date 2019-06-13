import json

fpat = r"D:/data/biendata/ccks2019_el/ccks_train_data/develop.hanlp.filter.json"
spat = r"D:/data/biendata/ccks2019_el/ccks_train_data/develop.hanlp.filter.str.json"

rd = open(fpat, "r", encoding="utf-8")
wt = open(spat, "w", encoding="utf-8")
for line in rd:
    jline = json.loads(line)
    njline = jline
    mention_data = []
    for mention in jline["mention_data"]:
        mention["offset"] = str(mention["offset"])
        mention_data.append(mention)
    njline["mention_data"] = mention_data
    wt.write(json.dumps(njline, ensure_ascii=False))
    wt.write("\n")
    wt.flush()

wt.close()


