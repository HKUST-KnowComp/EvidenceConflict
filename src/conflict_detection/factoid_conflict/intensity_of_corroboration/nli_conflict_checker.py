import os
import torch

from tqdm import tqdm
from get_data import filter_by_nli
from utils import load_json, write_json
from transformers import pipeline


def nli_check(question, text0, text1, classifier):

    orig_score = classifier(
        {"text": question + "? " + text0, "text_pair": question + "? " + text1}, top_k=3
    )
    score = {k["label"]: k["score"] for k in orig_score}
    unconditioned = {
        "c>e": int(score["CONTRADICTION"] > score["ENTAILMENT"]),
        "max": int(orig_score[0]["label"] == "CONTRADICTION"),
    }

    return unconditioned


def nli_detection(num_facts, eva_nli_models):
    for eva_nli_model in eva_nli_models:

        in_fn = f"data/strategyqa_shuffled_{num_facts}facts-overlapped-checked.json"
        out_fn = f"test/strategyqa_{num_facts}facts({eva_nli_model.replace('microsoft/','')}).json"

        print(f"eva_nli_model is {eva_nli_model}")
        classifier = pipeline(
            "text-classification", model=eva_nli_model, device="cuda:0"
        )

        data = load_json(in_fn)

        for i, instance in enumerate(tqdm(data)):
            score_list = {}

            question = instance["question"]
            evidence_list = instance["evidence"]
            original_evidence = evidence_list["original"]
            nli_scores = instance["nli-check"]
            res = filter_by_nli(nli_scores["original"])

            # overlap-1:
            res1 = filter_by_nli(nli_scores["overlap-1"])
            if res == True and res1 == True:
                overlap_evidence = evidence_list["overlap-1"]
                score_list["overlap-1"] = [
                    nli_check(
                        question, original_evidence, overlap_evidence, classifier
                    ),
                    nli_check(
                        question, overlap_evidence, original_evidence, classifier
                    ),
                ]

            # overlap-2:
            res2 = filter_by_nli(nli_scores["overlap-2"])
            if res == True and res2 == True:
                overlap_evidence = evidence_list["overlap-2"]
                score_list["overlap-2"] = [
                    nli_check(
                        question, original_evidence, overlap_evidence, classifier
                    ),
                    nli_check(
                        question, overlap_evidence, original_evidence, classifier
                    ),
                ]

            if num_facts == 4:
                # overlap-3:
                res3 = filter_by_nli(nli_scores["overlap-3"])
                if res == True and res3 == True:
                    overlap_evidence = evidence_list["overlap-3"]
                    score_list["overlap-3"] = [
                        nli_check(
                            question, original_evidence, overlap_evidence, classifier
                        ),
                        nli_check(
                            question, overlap_evidence, original_evidence, classifier
                        ),
                    ]

            instance[eva_nli_model.replace("microsoft/", "")] = score_list

            if i % 10 == 0:
                write_json(out_fn, data)

        write_json(out_fn, data)


if __name__ == "__main__":

    nli_detection(3, ["microsoft/deberta-v2-xxlarge-mnli"])
