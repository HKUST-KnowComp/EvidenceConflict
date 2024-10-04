from tqdm import tqdm
from utils import load_json, write_json
from transformers import pipeline


def is_conflict(question, text0, text1, classifier):

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


def nli_detection(num_facts, num_modified_facts, if_shuffle, eva_nli_models):
    for eva_nli_model in eva_nli_models:

        in_fn = f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{if_shuffle}shuffle-test_set.json"
        out_fn = f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{if_shuffle}shuffle({eva_nli_model.replace('microsoft/','')}).json"

        print(f"eva_nli_model is {eva_nli_model}")
        classifier = pipeline(
            "text-classification", model=eva_nli_model, device="cuda:0"
        )

        data = load_json(in_fn)

        for i, instance in enumerate(tqdm(data)):
            question = instance["question"]
            evidence = instance["evidence"]
            ori_evidence = instance["original_evidence"]

            instance[eva_nli_model.replace("microsoft/", "")] = [
                is_conflict(question, ori_evidence, evidence, classifier),
                is_conflict(question, evidence, ori_evidence, classifier),
            ]

            if i % 10 == 0:
                write_json(out_fn, data)

        write_json(out_fn, data)


if __name__ == "__main__":

    nli_detection(4, 2, 1, ["microsoft/deberta-v2-xxlarge-mnli"])
