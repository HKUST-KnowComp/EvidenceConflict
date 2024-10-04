import os, re
import random
from tqdm import tqdm
from transformers import pipeline
from utils import load_json, write_json, get_llm_response


def filter_dataset_by_facts(num_fact_threshold):
    in_fn = f"data/{num_fact_threshold}facts/strategyqa_train.json"
    out_fn = f"data/{num_fact_threshold}facts/strategyqa_{num_fact_threshold}facts.json"  # only have 4 facts

    data_set = load_json(in_fn)

    include_idx = []
    for i in tqdm(range(len(data_set))):
        if len(data_set[i]["facts"]) == num_fact_threshold:
            include_idx.append(i)
    data_set = [data_set[id_] for id_ in include_idx]

    def filter_dict(old_dict):
        new_dict = {
            key: old_dict[key] for key in ["qid", "question", "answer", "facts"]
        }
        return new_dict

    data_set = [filter_dict(i) for i in data_set]
    write_json(out_fn, data_set)


modify_fact_prompt = """Modify the statement to suggest otherwise that contradicts the original:

Statement: A pound sterling is fiat money.
Modified statement (in JSON format): {{"modified statement": "A pound sterling is a kind of cryptocurrency."}}

Statement: Dogs have sensitive ears that can hear as far as a quarter of a mile away.
Modified statement (in JSON format): {{"modified statement": "Dogs have average hearing abilities and cannot hear beyond a few yards."}}

Statement: Relay races are athletic track and field events.
Modified statement (in JSON format): {{"modified statement": "Relay races are intellectual board games."}}

Statement: {}
Modified statement (in JSON format):"""


def modify_fact(fact, modify_model):
    res = get_llm_response(modify_fact_prompt.format(fact), model=modify_model)
    retr_ct = 0
    while retr_ct <= 3:
        try:
            clean_res = eval(re.findall(r"{[\s\S]*?}", res)[0])
            res = clean_res["modified statement"]
            break

        except:
            retr_ct += 1
            print("retrying...", retr_ct)

        if retr_ct == 4:
            print("failed...")
            print(res)

    return res


def get_modified_facts(num, modify_model):

    in_fn = f"data/{num}facts/strategyqa_{num}facts.json"
    out_fn = f"data/{num}facts/strategyqa_modified_{num}facts.json"

    # if os.path.exists(out_fn):
    #     collected_data = load_json(out_fn)
    # else:
    collected_data = []

    data_set = load_json(in_fn)

    for i in tqdm(range(len(collected_data), len(data_set))):
        item = data_set[i]
        modified_facts = []
        for fact in item["facts"]:
            modified_facts.append(modify_fact(fact, modify_model))
        item["facts"] = {"original": item["facts"], "modified": modified_facts}
        collected_data.append(item)

        if i % 10 == 0:
            write_json(out_fn, collected_data)
    write_json(out_fn, collected_data)


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


def get_mixed_facts_evidence(num_facts, num_modified_facts, gen_model, if_shuffle):
    in_fn = f"data/{num_facts}facts/strategyqa_modified_{num_facts}facts.json"
    out_fn = f"data/{num_facts}facts/strategyqa_{num_facts}facts_evi_{num_modified_facts}mofified_{if_shuffle}shuffle.json"
    data = load_json(in_fn)

    if num_modified_facts == 0:

        for i, item in enumerate(tqdm(data)):
            facts = item["facts"]
            original_facts = facts["original"]
            modified_facts = facts["modified"]
            mixed_facts = original_facts
            facts["mixed_facts"] = mixed_facts

            if if_shuffle == 0:
                item["evidence"] = generate_evidence(mixed_facts, gen_model)
            elif if_shuffle == 1:
                random.seed(2023)
                random.shuffle(mixed_facts)
                facts["shuffled_facts"] = mixed_facts
                item["evidence"] = generate_evidence(mixed_facts, gen_model)

            if i % 10 == 0:
                write_json(out_fn, data)
        write_json(out_fn, data)

    if num_modified_facts == 1:

        random.seed(2023)
        rand_list = []
        n = len(data)
        for i in range(n):
            rand_list.append(random.randint(0, num_facts - 1))

        for i, item in enumerate(tqdm(data)):
            facts = item["facts"]
            original_facts = facts["original"]
            modified_facts = facts["modified"]

            pick_idx = rand_list[i]
            mixed_facts = []
            for ix in range(pick_idx):
                mixed_facts.append(original_facts[ix])
            mixed_facts.append(modified_facts[pick_idx])
            for ix in range(pick_idx + 1, num_facts):
                mixed_facts.append(original_facts[ix])

            facts["mixed_facts"] = mixed_facts

            if if_shuffle == 0:
                item["evidence"] = generate_evidence(mixed_facts, gen_model)
            elif if_shuffle == 1:
                random.seed(2023)
                random.shuffle(mixed_facts)
                facts["shuffled_facts"] = mixed_facts
                item["evidence"] = generate_evidence(mixed_facts, gen_model)

            if i % 10 == 0:
                write_json(out_fn, data)
        write_json(out_fn, data)

    if num_modified_facts == 2:

        random.seed(2023)
        rand_list = []
        n = len(data)
        for i in range(n):
            a = random.randint(0, num_facts - 1)

            done = False
            while done == False:
                b = random.randint(0, num_facts - 1)
                if a != b:
                    done = True
                else:
                    done = False

            rand_list.append(sorted([a, b]))

        for i, item in enumerate(tqdm(data)):
            facts = item["facts"]
            original_facts = facts["original"]
            modified_facts = facts["modified"]

            pick_idxs = rand_list[i]
            mixed_facts = []
            for ix in range(pick_idxs[0]):
                mixed_facts.append(original_facts[ix])
            mixed_facts.append(modified_facts[pick_idxs[0]])
            for ix in range(pick_idxs[0] + 1, pick_idxs[1]):
                mixed_facts.append(original_facts[ix])
            mixed_facts.append(modified_facts[pick_idxs[1]])
            for ix in range(pick_idxs[1] + 1, num_facts):
                mixed_facts.append(original_facts[ix])

            facts["mixed_facts"] = mixed_facts

            if if_shuffle == 0:
                item["evidence"] = generate_evidence(mixed_facts, gen_model)
            elif if_shuffle == 1:
                random.seed(2023)
                random.shuffle(mixed_facts)
                facts["shuffled_facts"] = mixed_facts
                item["evidence"] = generate_evidence(mixed_facts, gen_model)

            if i % 10 == 0:
                write_json(out_fn, data)
        write_json(out_fn, data)

    if num_facts == num_modified_facts:
        for i, item in enumerate(tqdm(data)):
            facts = item["facts"]
            original_facts = facts["original"]
            modified_facts = facts["modified"]
            facts["mixed_facts"] = modified_facts
            mixed_facts = modified_facts

            if if_shuffle == 0:
                item["evidence"] = generate_evidence(mixed_facts, gen_model)
            elif if_shuffle == 1:
                random.seed(2023)
                random.shuffle(mixed_facts)
                facts["shuffled_facts"] = mixed_facts
                item["evidence"] = generate_evidence(mixed_facts, gen_model)

            if i % 10 == 0:
                write_json(out_fn, data)
        write_json(out_fn, data)

    if num_modified_facts == 3 and num_facts == 4:

        random.seed(2023)
        rand_list = []
        n = len(data)
        for i in range(n):
            rand_list.append(random.randint(0, num_facts - 1))

        for i, item in enumerate(tqdm(data)):
            facts = item["facts"]
            original_facts = facts["original"]
            modified_facts = facts["modified"]

            pick_idx = rand_list[i]
            mixed_facts = []
            for ix in range(pick_idx):
                mixed_facts.append(modified_facts[ix])
            mixed_facts.append(original_facts[pick_idx])
            for ix in range(pick_idx + 1, num_facts):
                mixed_facts.append(modified_facts[ix])

            facts["mixed_facts"] = mixed_facts

            if if_shuffle == 0:
                item["evidence"] = generate_evidence(mixed_facts, gen_model)
            elif if_shuffle == 1:
                random.seed(2023)
                random.shuffle(mixed_facts)
                facts["shuffled_facts"] = mixed_facts
                item["evidence"] = generate_evidence(mixed_facts, gen_model)

            if i % 10 == 0:
                write_json(out_fn, data)
        write_json(out_fn, data)


def consistency_check(num_facts, num_modified_facts, if_shuffle, check_nli_model):
    in_fn = f"data/{num_facts}facts/strategyqa_{num_facts}facts_evi_{num_modified_facts}mofified_{if_shuffle}shuffle.json"
    out_fn = f"data/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{if_shuffle}shuffle-checked.json"

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

    def nli_check_evidence(text, seed_facts, opposite_facts):
        """We check
        (1) whether the evidence entails its seed evidence used in generation, and
        (2) whether the evidence contradicts its opposite evidence (counterparts in the modified list).
        """
        return {
            "support": nli_batched([(text, fact) for fact in seed_facts]),
            "opposite": nli_batched([(text, fact) for fact in opposite_facts]),
        }

    if if_shuffle == 0:
        for i, item in enumerate(tqdm(data)):
            evidence = item["evidence"]

            original_facts = item["facts"]["original"]
            modified_facts = item["facts"]["modified"]
            seed_facts = item["facts"]["mixed_facts"]

            opposite_facts = []
            for ix in range(len(seed_facts)):
                if seed_facts[ix] == original_facts[ix]:
                    opposite_facts.append(modified_facts[ix])
                elif seed_facts[ix] == modified_facts[ix]:
                    opposite_facts.append(original_facts[ix])

            item["nli-check"] = nli_check_evidence(evidence, seed_facts, opposite_facts)

        if i % 10 == 0:
            write_json(out_fn, data)
        write_json(out_fn, data)

    if if_shuffle == 1:
        for i, item in enumerate(tqdm(data)):
            evidence = item["evidence"]

            original_facts = item["facts"]["original"]
            modified_facts = item["facts"]["modified"]
            shuffled_facts = item["facts"]["shuffled_facts"]
            seed_facts = shuffled_facts

            opposite_facts = []
            for ix in range(len(seed_facts)):
                for jx in range(len(seed_facts)):
                    if seed_facts[ix] == original_facts[jx]:
                        opposite_facts.append(modified_facts[jx])
                    elif seed_facts[ix] == modified_facts[jx]:
                        opposite_facts.append(original_facts[jx])

            item["nli-check"] = nli_check_evidence(evidence, seed_facts, opposite_facts)

        if i % 10 == 0:
            write_json(out_fn, data)
        write_json(out_fn, data)


if __name__ == "__main__":
    """modify facts"""
    # modify_model = 'claude-v3-sonnet'
    # get_modified_facts(3, modify_model)
    # get_modified_facts(4, modify_model)

    """generate evidence"""
    gen_model = "claude-v3-sonnet"
    # num_facts = 4
    # if_shuffle = 0
    get_mixed_facts_evidence(3, 3, gen_model, 0)
    get_mixed_facts_evidence(3, 3, gen_model, 1)
    # for if_shuffle in [0,1]:
    #     get_mixed_facts_evidence(num_facts, 1, gen_model, if_shuffle)
    # for if_shuffle in [0,1]:
    #     get_mixed_facts_evidence(num_facts, 3, gen_model, if_shuffle)
    # get_mixed_facts_evidence(4, 1, gen_model, 0)
    # get_mixed_facts_evidence(4, 3, gen_model, 0)

    # '''check consistency'''
    # check_nli_model = "microsoft/deberta-v2-xxlarge-mnli"

    # consistency_check(4, 0, 0, check_nli_model)
    # for if_shuffle in [0,1]:
    #     consistency_check(4, 1, if_shuffle, check_nli_model)
    #     consistency_check(4, 3, if_shuffle, check_nli_model)
    #     consistency_check(4, 4, if_shuffle, check_nli_model)
