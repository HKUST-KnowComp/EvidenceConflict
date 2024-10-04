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


def as_detection(dataset, length, reverse, eva_as_models):
    for eva_as_model in eva_as_models:

        in_fn = f"data/{dataset}-{length}-triples-alter_answer.json"
        out_fn = f"data/{dataset}-{length}({eva_as_model}){reverse}.json"

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

        if len(data) > 500:
            data = data[:500]

        for i, instance in enumerate(tqdm(data)):
            texts = instance["texts"]
            revision = instance["revision"]
            question = instance["question"]
            within_texts = [
                (texts[0], texts[1]),
                (texts[0], texts[2]),
                (texts[1], texts[2]),
            ]
            if reverse == "":
                instance["within_texts"] = [
                    is_conflict_as(question, x, y, scorer, eva_as_model)
                    for x, y in within_texts
                ]
            elif reverse == "_reverse":
                instance["within_texts"] = [
                    is_conflict_as(question, y, x, scorer, eva_as_model)
                    for x, y in within_texts
                ]

            if revision:
                revision_texts = [
                    (revision, texts[0]),
                    (revision, texts[1]),
                    (revision, texts[2]),
                ]
                if reverse == "":
                    instance["revision_texts"] = [
                        is_conflict_as(question, x, y, scorer, eva_as_model)
                        for x, y in revision_texts
                    ]
                if reverse == "_reverse":
                    instance["revision_texts"] = [
                        is_conflict_as(question, y, x, scorer, eva_as_model)
                        for x, y in revision_texts
                    ]
            else:
                instance["revision_texts"] = None

            if i % 10 == 0:
                write_json(out_fn, data)

        write_json(out_fn, data)


def minicheck_detection(dataset, length, eva_mini_models):
    for eva_model in eva_mini_models:
        in_fn = f"data/{dataset}-{length}-triples-alter_answer.json"
        out_fn = f"data/{dataset}-{length}({eva_model})_reverse.json"

        print(f"eva_mini_model is {eva_model}")

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

        if len(data) > 500:
            data = data[:500]

        for i, instance in enumerate(tqdm(data)):
            if "within_texts" in instance or "revision_texts" in instance:
                pass

            else:
                texts = instance["texts"]
                revision = instance["revision"]
                question = instance["question"]
                within_texts = [
                    (texts[0], texts[1]),
                    (texts[0], texts[2]),
                    (texts[1], texts[2]),
                ]
                # instance['within_texts'] = [is_conflict_mini(question, x, y, scorer, eva_model) for x,y in within_texts]
                instance["within_texts"] = [
                    is_conflict_mini(question, y, x, scorer, eva_model)
                    for x, y in within_texts
                ]
                if revision:
                    revision_texts = [
                        (revision, texts[0]),
                        (revision, texts[1]),
                        (revision, texts[2]),
                    ]
                    # instance['revision_texts'] = [is_conflict_mini(question, x, y, scorer, eva_model) for x,y in revision_texts]
                    instance["revision_texts"] = [
                        is_conflict_mini(question, y, x, scorer, eva_model)
                        for x, y in revision_texts
                    ]
                else:
                    instance["revision_texts"] = None

                if i % 10 == 0:
                    write_json(out_fn, data)

            write_json(out_fn, data)


if __name__ == "__main__":
    # for dataset in ["nq_open","cwq"]:
    #     for length in ["short","long"]:
    #         minicheck_detection(dataset, length, ['roberta-large', 'deberta-v3-large', 'flan-t5-large'])\
    as_detection("nq_open", "short", "", ["as-base", "as-large"])
