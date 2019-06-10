import json

from kashgari.tasks.seq_labeling import BLSTMCRFModel
from tqdm import tqdm

from ccks_ner import dload


def eval_pre():
    dic_path = "D:/data/biendata/ccks2019_el/ccks_train_data/entity.dic.txt"
    dic = open(dic_path, "r", encoding="utf-8").readlines()

    key_file_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\validate.json"
    # result_file_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\validate.json"
    # result_file_path = r"D:/data/biendata/ccks2019_el/ccks_train_data/validate.json.pre.json"
    result_file_path = r"D:/data/biendata/ccks2019_el/ccks_train_data/validate.json.hanlp.al4.dic.pre"

    """
    f:0.674604976995421
    p:0.6998009112286925
    r:0.6875935332157583   
    
    f:0.4904296980722611
    p:0.3846579728313432
    r:0.78017291967292
    
    f:0.4905715662515988
    p:0.38472297270998246
    r:0.7808581048581054
    
    """

    def load_mention_ids(file_path):
        return [
            [mention["mention"] for mention in json.loads(line)["mention_data"]]
            # [mention["kb_id"] for mention in json.loads(line)["mention_data"]]
            for line in open(file_path, 'r', encoding="utf-8").readlines()
        ]

    def load_key_mention_ids(file_path):
        return [
            ["UKN" if mention["kb_id"] == "NIL" else mention["mention"] for mention in json.loads(line)["mention_data"]]
            # [mention["kb_id"] for mention in json.loads(line)["mention_data"]]
            for line in open(file_path, 'r', encoding="utf-8").readlines()
        ]

    key_mention_ids = load_key_mention_ids(key_file_path)
    # key_mention_ids = [x for x in key_mention_ids if x != 'UKN']
    key_mention_ids = [list(filter(lambda k: k is not 'UKN', l)) for l in key_mention_ids]
    result_mention_ids = load_mention_ids(result_file_path)
    result_mention_ids = [list(filter(lambda k: k is not 'UKN', l)) for l in result_mention_ids]
    # result_mention_ids = list(filter(lambda k: k is not '', result_mention_ids))

    f_all = 0.0
    p_all = 0.0
    r_all = 0.0

    for ind, key_mention in enumerate(tqdm(key_mention_ids)):

        result_mention = set(result_mention_ids[ind])
        result_mention.add("NIL")
        result_mention.remove("NIL")

        key_mention = set(key_mention)
        key_mention.add("NIL")
        key_mention.remove("NIL")

        for k_m in key_mention:
            if k_m not in result_mention:
                # if k_m in dic:
                    print(k_m)

        mention_intersection = list(result_mention.intersection(key_mention))
        p = 1.0
        r = 1.0
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

    print("f:{}".format(f_mean))
    print("p:{}".format(p_mean))
    print("r:{}".format(r_mean))


def eval_crf():
    model_path = r"D:\data\biendata\ccks2019_el\ner_model"
    model = BLSTMCRFModel.load_model(model_path)
    validate_x, validate_y = dload.load_json_data('validate')
    model.evaluate(validate_x, validate_y)


def main():
    # eval_crf()
    eval_pre()


if __name__ == '__main__':
    main()
