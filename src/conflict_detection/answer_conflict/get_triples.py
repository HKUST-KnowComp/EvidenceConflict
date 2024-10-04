from utils import load_json, write_json
from tqdm import tqdm
from itertools import product
from random import sample, shuffle, seed, choice


def get_triples_set(dataset, length):
    in_fn = f"data/{dataset}-{length}-checked.json"
    out_fn = f"data/{dataset}-{length}-triples.json"

    data = load_json(in_fn)

    candidates = []
    seed(2023)

    for i, item in enumerate(tqdm(data)):

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
        gt_answers = [a for a in item["groundtruth_answer"] if a in answer2evid]
        possible_answers = [a for a in item["possible_answer"] if a in answer2evid]
        possible_but_not_gt_answers = [
            a for a in possible_answers if a not in gt_answers
        ]

        all_answers = list(set(gt_answers).union(set(possible_answers)))

        # same answer evids...
        for answer in answer2evid:
            evids = answer2evid[answer]
            if len(evids) > 1:
                try:
                    if answer in item["groundtruth_answer"]:
                        other_answer = choice(possible_but_not_gt_answers)
                    else:
                        # select from all other answers
                        other_answer = choice([a for a in all_answers if a != answer])

                    evids.append(answer2evid[other_answer][0])
                    candidates.append(
                        {
                            "question": item["question"],
                            "texts": evids,
                            "answers": [answer, answer, other_answer],
                            "groundtruth": item["groundtruth_answer"],
                            "all_answers": all_answers,
                            "instance_id": i,
                        }
                    )
                except Exception as e:
                    print(e)
                    continue

    seed(2023)
    shuffle(candidates)
    print("Candidates:", len(candidates))


if __name__ == "__main__":
    for dataset in ["cwq", "nq_open"]:
        for length in ["long", "short"]:
            print(dataset, length)
            get_triples_set(dataset, length)
