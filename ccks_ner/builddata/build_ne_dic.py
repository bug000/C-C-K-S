import json

kg_path = r"D:/data/biendata/ccks2019_el/ccks_train_data/kb_data"
dic_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\entity.fdic.txt"
dic_contain_space_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\entity.cspace.fdic.txt"

kg_reader = open(kg_path, "r", encoding="utf-8")
cont_set = set()
cont_contain_space_set = set()

for kg_line in kg_reader:
    entity = json.loads(kg_line)
    for subject_alias in entity["alias"]:
        if subject_alias not in cont_set:
            if " " in subject_alias:
                cont_contain_space_set.add(subject_alias+"\n")
            else:
                # cont_set.add(subject_alias+"\talias\t9\n")
                # cont_set.add(subject_alias+"\n")
                # if len(subject_alias) > 5:
                #     cont_set.add(subject_alias + "\t10000\n")
                    # cont_set.add(subject_alias + "\n")
                # else:
                    cont_set.add(subject_alias + "\n")

    subject = entity["subject"]

    if subject not in cont_set:
        if " " in subject:
            cont_contain_space_set.add(subject+"\n")
        else:
            # if len(subject) > 5:
            #     cont_set.add(subject + "\t10000\n")
                # cont_set.add(subject + "\n")
            # else:
                cont_set.add(subject + "\n")
            # cont_set.add(subject+"\n")
            # cont_set.add(subject + "\t100\tsubject\n")
            # cont_set.add(subject+"\tsubject\t10\n")

dic_wt = open(dic_path, "w", encoding="utf-8")
dic_wt.writelines(cont_set)
dic_wt.close()

dic_wt = open(dic_contain_space_path, "w", encoding="utf-8")
dic_wt.writelines(cont_contain_space_set)
dic_wt.close()
