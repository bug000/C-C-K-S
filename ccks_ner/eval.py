import json

from kashgari.tasks.seq_labeling import BLSTMCRFModel

from ccks_ner import dload


def eval_pre():

    key_file_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\validate.json"
    # result_file_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\validate.json"
    result_file_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\validate.pre.nil.json"

    def load_mention_ids(file_path):
        return [
            # [mention["mention"] for mention in json.loads(line)["mention_data"]]
            [mention["kb_id"] for mention in json.loads(line)["mention_data"]]
            for line in open(file_path, 'r', encoding="utf-8").readlines()
        ]

    key_mention_ids = load_mention_ids(key_file_path)
    result_mention_ids = load_mention_ids(result_file_path)

    f_all = 0.0
    p_all = 0.0
    r_all = 0.0

    for ind, key_mention in enumerate(key_mention_ids):

        result_mention = set(result_mention_ids[ind])
        result_mention.add("NIL")
        result_mention.remove("NIL")

        key_mention = set(key_mention)
        key_mention.add("NIL")
        key_mention.remove("NIL")

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

    print(f_mean)


def eval_crf():
    model_path = r"D:\data\biendata\ccks2019_el\ner_model"
    model = BLSTMCRFModel.load_model(model_path)
    validate_x, validate_y = dload.load_json_data('validate')
    model.evaluate(validate_x, validate_y)


def main():
    eval_crf()
    # eval_pre()


if __name__ == '__main__':
    main()
