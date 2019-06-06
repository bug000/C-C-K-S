from sklearn.model_selection import train_test_split


def save_clf_data(lines: list, data_type: str):
    data_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\clf_data\split\{}.tsv"
    with open(data_dir.format(data_type), 'w', encoding='utf-8') as data_wt:
        data_wt.writelines(lines)


all_clf_data_path = r"D:/data/biendata/ccks2019_el/ccks_train_data/clf_data/all.text.tsv"

all_clf_data = open(all_clf_data_path, "r", encoding="utf-8").readlines()
inds = [line.split("\t")[0] for line in all_clf_data]

inds_train, inds_o, lines_train, lines_o = train_test_split(inds, all_clf_data, test_size=0.3)

inds_val, inds_test, lines_val, lines_test = train_test_split(inds_o, lines_o, test_size=0.5)

save_clf_data(lines_train, "train")
save_clf_data(lines_val, "validate")
save_clf_data(lines_test, "test")
