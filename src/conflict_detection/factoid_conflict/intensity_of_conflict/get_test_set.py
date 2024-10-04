from utils import load_json, write_json
from tqdm import tqdm
from itertools import combinations
from random import sample, shuffle, seed
from collections import Counter

""" Obtain the test set, the data are filtered (generated evidence should entail all their facts) 

    - Each instance should pass the consistency check, i.e., it should ENTAILs all its seed facts, and CONTRADICTs all the alternative facts.
    - We only keep the evidence that meet the above requirements.
"""


def filter_by_nli(nli_results):
    # if all([item[0]['label'] == 'ENTAILMENT' for item in nli_results]):
    if all(
        [item[0]["label"] == "ENTAILMENT" for item in nli_results["support"]]
    ) and all(
        [item[0]["label"] == "CONTRADICTION" for item in nli_results["opposite"]]
    ):

        return True
    else:
        return False


def get_set_for_test(num_facts, num_modified_facts, if_shuffle):
    in_fn = f"data/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{if_shuffle}shuffle-checked.json"
    out_fn = f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{if_shuffle}shuffle-test_set.json"

    original_data = load_json(
        f"data/{num_facts}facts/strategyqa_{num_facts}facts_0mofified_0shuffle-checked.json"
    )
    data = load_json(in_fn)
    filtered_test_set = []

    for id_, item in enumerate(tqdm(data)):
        original_nli_scores = original_data[id_]["nli-check"]
        nli_scores = item["nli-check"]

        res1 = filter_by_nli(original_nli_scores)
        res2 = filter_by_nli(nli_scores)
        if res1 == True and res2 == True:
            item["original_evidence"] = original_data[id_]["evidence"]
            item["id_instance"] = id_
            del item["nli-check"]

            filtered_test_set.append(item)

        if id_ % 10 == 0:
            write_json(out_fn, filtered_test_set)
    write_json(out_fn, filtered_test_set)
    print(f"qualified candidates: {len(filtered_test_set)} pairs")


if __name__ == "__main__":
    for num_modified_facts in range(1, 5):
        for if_shuffle in [0, 1]:
            get_set_for_test(4, num_modified_facts, if_shuffle)
