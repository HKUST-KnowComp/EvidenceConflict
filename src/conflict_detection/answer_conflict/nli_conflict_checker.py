import os
from utils import load_json, write_json
from transformers import pipeline
from tqdm import tqdm


def is_conflict_nli(question, text0, text1, classifier):

    orig_score = classifier({"text": text0, "text_pair": text1}, top_k=3)
    score = {k["label"]: k["score"] for k in orig_score}
    unconditioned = {
        "c>e": int(score["CONTRADICTION"] > score["ENTAILMENT"]),
        "max": int(orig_score[0]["label"] == "CONTRADICTION"),
    }

    orig_score = classifier(
        {"text": question + "? " + text0, "text_pair": question + "? " + text1}, top_k=3
    )
    score = {k["label"]: k["score"] for k in orig_score}
    conditioned = {
        "c>e": int(score["CONTRADICTION"] > score["ENTAILMENT"]),
        "max": int(orig_score[0]["label"] == "CONTRADICTION"),
    }

    return {"unconditioned": unconditioned, "conditioned": conditioned}


def nli_detection(dataset, length, eva_nli_models):
    for eva_nli_model in eva_nli_models:

        in_fn = f"data/{dataset}-{length}-triples-alter_answer.json"
        out_fn = (
            f"data/{dataset}-{length}({eva_nli_model.replace('/','-')})_reverse.json"
        )

        print(f"eva_nli_model is {eva_nli_model}")
        classifier = pipeline(
            "text-classification", model=eva_nli_model, device="cuda:0"
        )

        if os.path.exists(out_fn):
            data = load_json(out_fn)
        else:
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
                # instance['within_texts'] = [is_conflict_nli(question, x, y, classifier) for x,y in within_texts]
                instance["within_texts"] = [
                    is_conflict_nli(question, y, x, classifier) for x, y in within_texts
                ]
                if revision:
                    revision_texts = [
                        (revision, texts[0]),
                        (revision, texts[1]),
                        (revision, texts[2]),
                    ]
                    # instance['revision_texts'] = [is_conflict_nli(question, x, y, classifier) for x,y in revision_texts]
                    instance["revision_texts"] = [
                        is_conflict_nli(question, y, x, classifier)
                        for x, y in revision_texts
                    ]
                else:
                    instance["revision_texts"] = None

                if i % 10 == 0:
                    write_json(out_fn, data)

            write_json(out_fn, data)
