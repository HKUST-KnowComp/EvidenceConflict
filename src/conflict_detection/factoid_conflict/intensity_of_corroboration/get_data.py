import re
import random
from tqdm import tqdm
from transformers import pipeline
from utils import load_json, write_json, get_llm_response


def shuffle_facts(num_facts):
    in_fn = f"../test_set2_modified/data/{num_facts}facts/strategyqa_modified_{num_facts}facts.json"
    out_fn = f"data/strategyqa_shuffled_{num_facts}facts.json"

    data = load_json(in_fn)

    shuffle_orders = []
    random.seed(2023)
    for i in range(len(data)):
        if num_facts == 4:
            order = [0, 1, 2, 3]
        elif num_facts == 3:
            order = [0, 1, 2]
        else:
            print("wrong num_facts...")
        random.shuffle(order)
        shuffle_orders.append(order)

    for i, item in enumerate(tqdm(data)):
        facts = item["facts"]
        original_facts = facts["original"]
        modified_facts = facts["modified"]

        shuffled_ori, shuffled_mod = [], []
        ordertmp = shuffle_orders[i]

        for j in ordertmp:
            shuffled_ori.append(original_facts[j])
            shuffled_mod.append(modified_facts[j])

        facts["shuffled_original"] = shuffled_ori
        facts["shuffled_modified"] = shuffled_mod

        if i % 10 == 0:
            write_json(out_fn, data)

    write_json(out_fn, data)


evidence_prompt = """Keypoints: {}

Give me a paragraph of around 100 words using the keypoints (try to simulate the format of web search results):

Paragraph (should be in JSON format and formatted as {{"Paragraph": "TEXT"}}): """


def generate_evidence(facts, gen_evi_model):
    res = get_llm_response(
        evidence_prompt.format("\n- " + "\n- ".join(facts)), model=gen_evi_model
    )
    retr_ct = 0
    while retr_ct <= 3:
        try:
            clean_res = eval(re.findall(r"{[\s\S]*?}", res)[0])
            res = clean_res["Paragraph"]
            break

        except:
            retr_ct += 1
            print("retrying...", retr_ct)

        if retr_ct == 4:
            print("failed...")
            print(res)

    return res


def get_overlapped_facts_evidence(num_facts, gen_model):
    in_fn = f"data/strategyqa_shuffled_{num_facts}facts.json"
    out_fn = f"data/strategyqa_shuffled_{num_facts}facts-overlapped.json"
    data = load_json(in_fn)

    for i, item in enumerate(tqdm(data)):
        facts = item["facts"]
        shuffled_original = facts["shuffled_original"]
        shuffled_modified = facts["shuffled_modified"]

        evidence_list = {}
        # [O,O,O,O]
        evidence_list["original"] = generate_evidence(shuffled_original, gen_model)

        for overlap_num in range(1, num_facts):
            # [M,O], [M,O,O]...
            factstmp = [shuffled_modified[0]]
            for j in range(1, overlap_num + 1):
                factstmp.append(shuffled_original[j])
            facts[f"overlap-{overlap_num}"] = factstmp
            evidence_list[f"overlap-{overlap_num}"] = generate_evidence(
                factstmp, gen_model
            )

        item["evidence"] = evidence_list

        if i % 10 == 0:
            write_json(out_fn, data)
    write_json(out_fn, data)


def consistency_check(num_facts, check_nli_model):
    in_fn = f"data/strategyqa_shuffled_{num_facts}facts-overlapped.json"
    out_fn = f"data/strategyqa_shuffled_{num_facts}facts-overlapped-checked.json"

    data = load_json(in_fn)

    classifier = pipeline(
        "text-classification", model=check_nli_model, device="cuda:0", batch_size=32
    )

    def nli(e1, e2):
        return classifier({"text": e1, "text_pair": e2}, top_k=1)

    def nli_batched(list_of_e1_e2):
        return classifier(
            [{"text": e1, "text_pair": e2} for e1, e2 in list_of_e1_e2], top_k=1
        )

    def nli_check_evidence(text, support_facts, opposite_facts, neutral_facts=[]):
        if len(neutral_facts) > 0:
            return {
                "support": nli_batched([(text, fact) for fact in support_facts]),
                "opposite": nli_batched([(text, fact) for fact in opposite_facts]),
                "neutral": nli_batched([(text, fact) for fact in neutral_facts]),
            }
        else:
            return {
                "support": nli_batched([(text, fact) for fact in support_facts]),
                "opposite": nli_batched([(text, fact) for fact in opposite_facts]),
            }

    for i, item in enumerate(tqdm(data)):
        facts_list = item["facts"]
        evidence_list = item["evidence"]

        nli_check_list = {}

        # original_evidence [O1,O2,O3,O4] should support shuffled_original facts [O1,O2,O3,O4], and opposite [M]
        # support: [O1,O2,O3]; opposite: [M1]
        evidence = evidence_list["original"]
        support_facts = facts_list["shuffled_original"]
        opposite_facts = facts_list["shuffled_modified"][:1]

        nli_check_list["original"] = nli_check_evidence(
            evidence, support_facts, opposite_facts
        )

        # overlap-1 [M,O2] evidence should support overlap-1 facts [M,O2], opposite [O1], and neutral [O3,O4]
        # support: [M,O2]; opposite: [M1]; neutral: [O3]
        evidence = evidence_list["overlap-1"]
        support_facts = facts_list["overlap-1"]
        opposite_facts = facts_list["shuffled_original"][:1]
        neutral_facts = facts_list["shuffled_original"][2:]

        nli_check_list["overlap-1"] = nli_check_evidence(
            evidence, support_facts, opposite_facts, neutral_facts
        )

        # overlap-2 [M,O2,O3] evidence should support overlap-2 facts [M,O2,O3], opposite [O1], and neutral [O4]
        # support: [M,O2,O3]; opposite: [M1]
        evidence = evidence_list["overlap-2"]
        support_facts = facts_list["overlap-2"]
        opposite_facts = facts_list["shuffled_original"][:1]

        if num_facts == 3:
            nli_check_list["overlap-2"] = nli_check_evidence(
                evidence, support_facts, opposite_facts
            )
        else:
            neutral_facts = facts_list["shuffled_original"][3:4]

            nli_check_list["overlap-2"] = nli_check_evidence(
                evidence, support_facts, opposite_facts, neutral_facts
            )

            # overlap-3 [M,O2,O3,O4] evidence should support overlap-3 facts [M,O2,O3,O4], opposite [O1]
            evidence = evidence_list["overlap-3"]
            support_facts = facts_list["overlap-3"]
            opposite_facts = facts_list["shuffled_original"][:1]

            nli_check_list["overlap-3"] = nli_check_evidence(
                evidence, support_facts, opposite_facts
            )

        item["nli-check"] = nli_check_list

        if i % 10 == 0:
            write_json(out_fn, data)

    write_json(out_fn, data)


def filter_by_nli(nli_results):
    try:
        if (
            all([item[0]["label"] == "ENTAILMENT" for item in nli_results["support"]])
            and all(
                [
                    item[0]["label"] == "CONTRADICTION"
                    for item in nli_results["opposite"]
                ]
            )
            and all([item[0]["label"] == "NEUTRAL" for item in nli_results["neutral"]])
        ):

            return True
        else:
            return False

    except:
        if all(
            [item[0]["label"] == "ENTAILMENT" for item in nli_results["support"]]
        ) and all(
            [item[0]["label"] == "CONTRADICTION" for item in nli_results["opposite"]]
        ):

            return True
        else:
            return False


def get_set_for_test(num_facts):
    in_fn = f"data/strategyqa_shuffled_{num_facts}facts-overlapped-checked.json"
    out_fn = f"data/strategyqa_shuffled_{num_facts}facts-overlapped-checked-test.json"

    data = load_json(in_fn)
    filtered_test_set = []

    for id_, item in enumerate(tqdm(data)):
        nli_scores = item["nli-check"]

        res1 = filter_by_nli(nli_scores["original"])
        res2 = filter_by_nli(nli_scores["overlap-1"])
        res3 = filter_by_nli(nli_scores["overlap-2"])
        if num_facts == 4:
            res4 = filter_by_nli(nli_scores["overlap-3"])
        else:
            res4 = True

        if res1 == True and res2 == True and res3 == True and res4 == True:
            item["id_instance"] = id_
            del item["nli-check"], item["facts"]

            filtered_test_set.append(item)

        if id_ % 10 == 0:
            write_json(out_fn, filtered_test_set)

    write_json(out_fn, filtered_test_set)
    print(f"qualified candidates: {len(filtered_test_set)} pairs")


if __name__ == "__main__":
    """shuffle"""
    # for num_facts in [3,4]:
    #     shuffle_facts(num_facts)

    num_facts = 4
    """generate evidence"""
    gen_evi_model = "llama3-70b-instruct"
    get_overlapped_facts_evidence(num_facts, gen_evi_model)

    """consistency check"""
    check_nli_model = "microsoft/deberta-v2-xxlarge-mnli"
    consistency_check(num_facts, check_nli_model)

    """get test set"""
    get_set_for_test(num_facts)
