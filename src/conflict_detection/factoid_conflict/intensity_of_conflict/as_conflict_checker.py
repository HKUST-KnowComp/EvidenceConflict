import os
from alignscore import AlignScore
from utils import load_json, write_json
from tqdm import tqdm
from minicheck import MiniCheck


def is_conflict_as(question, text0, text1, scorer, eva_as_model):

    orig_score = scorer.score(contexts=[text0], claims=[text1])
    assert len(orig_score) == 1
    orig_score = 1 if orig_score[0] < 0.5 else 0
    orig_score = {eva_as_model: orig_score}

    prob = scorer.score(
        contexts=[question + "? " + text0], claims=[question + "? " + text1]
    )
    assert len(prob) == 1
    score = 1 if prob[0] < 0.5 else 0
    score = {eva_as_model: score}

    prob = {eva_as_model: prob}

    return {"unconditioned": orig_score, "conditioned": score, "conditioned_prob": prob}


def is_conflict_mini(question, text0, text1, scorer, eva_model):

    pred_label, raw_prob1, _, _ = scorer.score(docs=[text0], claims=[text1])
    assert len(pred_label) == 1
    orig_score = 1 if pred_label[0] == 0 else 0
    orig_score = {eva_model: orig_score}

    pred_label, raw_prob2, _, _ = scorer.score(
        docs=[question + "? " + text0], claims=[question + "? " + text1]
    )
    assert len(pred_label) == 1
    score = 1 if pred_label[0] == 0 else 0
    score = {eva_model: score}

    prob = {eva_model: raw_prob2}

    return {"unconditioned": orig_score, "conditioned": score, "conditioned_prob": prob}


def as_detection(num_facts, num_modified_facts, if_shuffle, eva_as_models):
    for eva_as_model in eva_as_models:

        in_fn = f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{if_shuffle}shuffle-test_set.json"
        out_fn = f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{if_shuffle}shuffle({eva_as_model}).json"

        print(f"eva_as_model is {eva_as_model}")
        if "base" in eva_as_model:
            scorer = AlignScore(
                model="roberta-base",
                batch_size=32,
                device="cuda:0",
                ckpt_path="/home/ubuntu/ConflictEvidence/AlignScore/AlignScore-base.ckpt",
                evaluation_mode="nli_sp",
            )
        elif "large" in eva_as_model:
            scorer = AlignScore(
                model="roberta-base",
                batch_size=32,
                device="cuda:0",
                ckpt_path="/home/ubuntu/ConflictEvidence/AlignScore/AlignScore-large.ckpt",
                evaluation_mode="nli_sp",
            )
        else:
            print("Wrong model name...")
            break

        data = load_json(in_fn)

        for i, instance in enumerate(tqdm(data)):
            question = instance["question"]
            evidence = instance["evidence"]
            ori_evidence = instance["original_evidence"]
            instance[eva_as_model] = [
                is_conflict_as(question, ori_evidence, evidence, scorer, eva_as_model),
                is_conflict_as(question, evidence, ori_evidence, scorer, eva_as_model),
            ]

            if i % 10 == 0:
                write_json(out_fn, data)

        write_json(out_fn, data)


def minicheck_detection(num_facts, num_modified_facts, if_shuffle, eva_mini_models):
    for eva_model in eva_mini_models:

        in_fn = f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{if_shuffle}shuffle-test_set.json"
        out_fn = f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{if_shuffle}shuffle({eva_model}).json"

        if "roberta" in eva_model:
            scorer = MiniCheck(
                model_name="roberta-large", device=f"cuda:0", cache_dir="./ckpts"
            )
        elif "deberta" in eva_model:
            scorer = MiniCheck(
                model_name="deberta-v3-large", device=f"cuda:0", cache_dir="./ckpts"
            )
        elif "flan" in eva_model:
            scorer = MiniCheck(
                model_name="flan-t5-large", device=f"cuda:0", cache_dir="./ckpts"
            )
        else:
            print("Wrong model name...")
            break

        data = load_json(in_fn)

        for i, instance in enumerate(tqdm(data)):
            question = instance["question"]
            evidence = instance["evidence"]
            ori_evidence = instance["original_evidence"]
            instance[eva_model] = [
                is_conflict_mini(question, ori_evidence, evidence, scorer, eva_model),
                is_conflict_mini(question, evidence, ori_evidence, scorer, eva_model),
            ]

            if i % 10 == 0:
                write_json(out_fn, data)

        write_json(out_fn, data)


if __name__ == "__main__":

    num_facts = 3
    for num_modified_facts in [1, 2, 3]:
        for if_shuffle in [0, 1]:
            minicheck_detection(
                num_facts,
                num_modified_facts,
                if_shuffle,
                ["roberta-large", "deberta-v3-large", "flan-t5-large"],
            )
    num_facts = 4
    for num_modified_facts in [1, 2, 3, 4]:
        for if_shuffle in [0, 1]:
            minicheck_detection(
                num_facts,
                num_modified_facts,
                if_shuffle,
                ["roberta-large", "deberta-v3-large", "flan-t5-large"],
            )
