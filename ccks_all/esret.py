from typing import List
from nltk.metrics.distance import jaccard_distance
from elasticsearch import Elasticsearch

es_servers = [{
    "host": "localhost",
    "port": "9200"
}]
es = Elasticsearch(es_servers)


def jaccard_sim(text1, text2):
    grams_reference = set(text1)  # 去重；如果不需要就改为list
    grams_model = set(text2)
    temp = 0
    for i in grams_reference:
        if i in grams_model:
            temp = temp + 1
    fenmu = len(grams_model) + len(grams_reference) - temp  # 并集
    jaccard_coefficient = float(temp / fenmu)  # 交集
    # jdis = jaccard_distance(grams_reference, grams_model)
    return jaccard_coefficient


def expand_sim_es(keywords: str, index_str: str = "subject_text", jaccard_filt=0.6) -> List:
    base_query = {
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "enti": ""
                        }
                    }
                ],
                "not": []
            }

        },
        "from": 0,
        "size": 50
    }
    base_query["query"]["bool"]["should"] = {"match": {"enti": keywords}}
    results = es.search(index=index_str, body=base_query)
    # results = json.dumps(results, ensure_ascii=False)
    hits = results["hits"]["hits"]
    explans = [hit["_source"]["enti"] for hit in hits]

    def filt_func(jaccard_val):
        # return jaccard_val > jaccard_filt and jaccard_val != 1
        return jaccard_val > jaccard_filt

    explans_jaccard_filt = list(filter(lambda exp_str: filt_func(jaccard_sim(keywords, exp_str)), explans))
    return explans_jaccard_filt


def main():
    # build_es_index()
    query_str = ""
    expand_sim_es(index_str="subject_text", keywords=query_str)


if __name__ == '__main__':
    main()
