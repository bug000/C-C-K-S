import json

from sklearn.model_selection import train_test_split


def save_data(data_dir: str, data_type: str, xs: list, ys: list):
    with open(data_dir.format(data_type), 'w', encoding="utf-8") as fsw:
        for ind, x_data_line in enumerate(xs):
            y_data_line = ys[ind]
            fsw.write(f"{''.join(x_data_line)}\t{''.join(y_data_line)}\n")

            if len(list(x_data_line)) != len(list(y_data_line)):
                print(x_data_line)


def save_json_data(data_dir: str, data_type: str, lines: list):
    with open(data_dir.format(data_type), 'w', encoding="utf-8") as fsw:
        for ind, json_line in enumerate(lines):
            fsw.write(json_line.strip() + "\n")


train_file_path = r"Z:/research/data/biendata/ccks2019_el/ccks2019_el/train.json"

data_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\{}.json"


inds = []
lines = []

with open(train_file_path, 'r', encoding='utf-8') as all_trains:
    for line_ind, json_line in enumerate(all_trains):
        inds.append(line_ind)
        lines.append(json_line)

inds_train, inds_o, lines_train, lines_o = train_test_split(inds, lines, test_size=0.3)

inds_val, inds_test, lines_val, lines_test = train_test_split(inds_o, lines_o, test_size=0.5)

# save_data(data_dir, "train", x_train, y_train)
# save_data(data_dir, "validate", x_val, y_val)
# save_data(data_dir, "test", x_test, y_test)

save_json_data(data_dir, "train", lines_train)
save_json_data(data_dir, "validate", lines_val)
save_json_data(data_dir, "test", lines_test)


