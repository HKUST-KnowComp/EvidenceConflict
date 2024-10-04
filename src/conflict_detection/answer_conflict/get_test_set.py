from utils import load_json, write_json
from tqdm import tqdm
from itertools import product
from random import sample, shuffle, seed


def get_pairs_for_test(dataset, length):
    data = load_json(f"{dataset}_valid-{length}-checked.json")

    candidates = []
    labels = []

    seed(2023)

    for item in tqdm(data):
        # reorganize
        answer_evidence = item["answer_evidence"]
        answer2evid = {}
        for answer, evids, lbls in answer_evidence:
            evids = [
                e
                for e, lbl in zip(evids, lbls)
                if lbl["nli"]["label"] == "ENTAILMENT"
                and lbl["consistency"]["label"] == "ENTAILMENT"
            ]
            if len(evids):
                answer2evid[answer] = evids
        # same answer evids...
        for answer in answer2evid:
            evids = answer2evid[answer]
            if len(evids) > 1:
                candidates.append(
                    {
                        "question": item["question"],
                        "texts": evids,
                        "answers": [answer, answer],
                        "groundtruth": item["groundtruth_answer"],
                        "has_conflict": 0,
                    }
                )
                labels.append(0)
        # different answer evids: between answer evids, only pick the first evidence
        gt_answers = [a for a in item["groundtruth_answer"] if a in answer2evid]
        possible_answers = [a for a in item["possible_answer"] if a in answer2evid]
        pairs = list(product(gt_answers, possible_answers)) + [
            p for p in product(possible_answers, possible_answers) if len(set(p)) > 1
        ]
        pairs = sample(pairs, min(3, len(pairs)))

        for a1, a2 in pairs:
            candidates.append(
                {
                    "question": item["question"],
                    "texts": [answer2evid[a1][0], answer2evid[a2][0]],
                    "answers": [a1, a2],
                    "groundtruth": item["groundtruth_answer"],
                    "has_conflict": 1,
                }
            )
            labels.append(1)

    seed(2023)
    shuffle(candidates)

    write_json(f"{dataset}_valid-{length}-checked-test.json", candidates)
