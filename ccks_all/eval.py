import json

from tqdm import tqdm


def load_mention_mess(file_path):
    all_men_texts = []
    all_men_ids = []
    for line in open(file_path, 'r', encoding="utf-8").readlines():
        json_line = json.loads(line)
        mention_data = json_line["mention_data"]
        mention_text = []
        mention_id = []
        for mention in mention_data:
            if mention["kb_id"] != "NIL":
                mention_text.append(mention["mention"])
                mention_id.append(mention["kb_id"])
        all_men_texts.append(mention_text)
        all_men_ids.append(mention_id)
    return all_men_texts, all_men_ids


def eval_dset(key_mention_ids, result_mention_ids):

    f_all = 0.0
    p_all = 0.0
    r_all = 0.0

    for ind, key_mention in enumerate(tqdm(key_mention_ids)):

        result_mention = set(result_mention_ids[ind])
        key_mention = set(key_mention)

        # for k_m in key_mention:
        #     if k_m not in subject_dict.keys():
                # if k_m not in result_mention:
                #     print(k_m)

        mention_intersection = list(result_mention.intersection(key_mention))
        p = 0.0
        r = 0.0
        if len(result_mention) != 0:
            p = len(mention_intersection) / len(result_mention)
        if len(key_mention) != 0:
            r = len(mention_intersection) / len(key_mention)

        f = 0.0
        if (p + r) != 0.0:
            f = (2 * p * r) / (p + r)

        f_all += f
        p_all += p
        r_all += r

    f_mean = f_all / len(key_mention_ids)
    p_mean = p_all / len(key_mention_ids)
    r_mean = r_all / len(key_mention_ids)

    print(f"f:{f_mean}")
    print(f"p:{p_mean}")
    print(f"r:{r_mean}")


# def eval_all_match(key_file, result_file):




def eval_file(key_file, result_file):
    key_men_texts, key_men_ids = load_mention_mess(key_file)
    re_men_texts, re_men_ids = load_mention_mess(result_file)

    print("text:")
    eval_dset(key_men_texts, re_men_texts)

    print("id:")
    eval_dset(key_men_ids, re_men_ids)


def main():
    key_file_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\test.json"
    result_file_path = r"D:/data/biendata/ccks2019_el/ccks_train_data/test.json.ngram.pre.json"
    eval_file(key_file_path, result_file_path)


if __name__ == '__main__':
    main()




