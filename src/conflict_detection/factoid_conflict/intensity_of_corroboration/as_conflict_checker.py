from tqdm import tqdm
from get_data import filter_by_nli
from minicheck import MiniCheck
from alignscore import AlignScore
from utils import load_json, write_json


def as_check(question, text0, text1, scorer):

    score = scorer.score(
        contexts=[question + "? " + text0], claims=[question + "? " + text1]
    )
    assert len(score) == 1
    score = 1 if score[0] < 0.5 else 0

    return score


def mini_check(question, text0, text1, scorer):

    pred_label, _, _, _ = scorer.score(
        docs=[question + "? " + text0], claims=[question + "? " + text1]
    )
    assert len(pred_label) == 1
    score = 1 if pred_label[0] == 0 else 0

    return score


"""overlap pairs"""
# def as_detection(num_facts, eva_as_models):
#     for eva_as_model in eva_as_models:

#         in_fn = f'data/strategyqa_shuffled_{num_facts}facts-overlapped-checked-test.json'
#         out_fn = f"test/strategyqa_{num_facts}facts({eva_as_model}).json"

#         print(f'eva_as_model is {eva_as_model}')
#         if 'base' in eva_as_model:
#             scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path='/home/ubuntu/ConflictEvidence/AlignScore/AlignScore-base.ckpt', evaluation_mode='nli_sp')
#         elif 'large' in eva_as_model:
#             scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path='/home/ubuntu/ConflictEvidence/AlignScore/AlignScore-large.ckpt', evaluation_mode='nli_sp')
#         else:
#             print('Wrong model name...')
#             break

#         data = load_json(in_fn)

#         for i, instance in enumerate(tqdm(data)):
#             score_list = {}

#             question = instance['question']
#             evidence_list = instance['evidence']
#             original_evidence = evidence_list['original']
#             nli_scores = instance['nli-check']
#             res = filter_by_nli(nli_scores['original'])

#             score_list['overlap-1'] = [as_check(question, original_evidence, overlap_evidence, scorer),as_check(question, overlap_evidence, original_evidence, scorer)]

#             score_list['overlap-2'] = [as_check(question, original_evidence, overlap_evidence, scorer),as_check(question, overlap_evidence, original_evidence, scorer)]

#             if num_facts == 4:
#                 core_list['overlap-3'] = [as_check(question, original_evidence, overlap_evidence, scorer),as_check(question, overlap_evidence, original_evidence, scorer)]

#             instance[eva_as_model] = score_list

#             if i % 10 == 0:
#                 write_json(out_fn, data)

#         write_json(out_fn, data)

"""single pairs"""


def as_detection(num_facts, eva_as_models):
    for eva_as_model in eva_as_models:

        in_fn = f"data/strategyqa_shuffled_{num_facts}facts-overlapped-checked.json"
        out_fn = f"test/strategyqa_{num_facts}facts({eva_as_model}).json"

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
                    as_check(question, original_evidence, overlap_evidence, scorer),
                    as_check(question, overlap_evidence, original_evidence, scorer),
                ]

            # overlap-2:
            res2 = filter_by_nli(nli_scores["overlap-2"])
            if res == True and res2 == True:
                overlap_evidence = evidence_list["overlap-2"]
                score_list["overlap-2"] = [
                    as_check(question, original_evidence, overlap_evidence, scorer),
                    as_check(question, overlap_evidence, original_evidence, scorer),
                ]

            if num_facts == 4:
                # overlap-3:
                res3 = filter_by_nli(nli_scores["overlap-3"])
                if res == True and res3 == True:
                    overlap_evidence = evidence_list["overlap-3"]
                    score_list["overlap-3"] = [
                        as_check(question, original_evidence, overlap_evidence, scorer),
                        as_check(question, overlap_evidence, original_evidence, scorer),
                    ]

            instance[eva_as_model] = score_list

            if i % 10 == 0:
                write_json(out_fn, data)

        write_json(out_fn, data)


def minicheck_detection(num_facts, eva_mini_models):
    for eva_model in eva_mini_models:

        in_fn = f"data/strategyqa_shuffled_{num_facts}facts-overlapped-checked.json"
        out_fn = f"test/strategyqa_{num_facts}facts({eva_model}).json"

        print(f"eva_model is {eva_model}")
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
                    mini_check(question, original_evidence, overlap_evidence, scorer),
                    mini_check(question, overlap_evidence, original_evidence, scorer),
                ]

            # overlap-2:
            res2 = filter_by_nli(nli_scores["overlap-2"])
            if res == True and res2 == True:
                overlap_evidence = evidence_list["overlap-2"]
                score_list["overlap-2"] = [
                    mini_check(question, original_evidence, overlap_evidence, scorer),
                    mini_check(question, overlap_evidence, original_evidence, scorer),
                ]

            if num_facts == 4:
                # overlap-3:
                res3 = filter_by_nli(nli_scores["overlap-3"])
                if res == True and res3 == True:
                    overlap_evidence = evidence_list["overlap-3"]
                    score_list["overlap-3"] = [
                        mini_check(
                            question, original_evidence, overlap_evidence, scorer
                        ),
                        mini_check(
                            question, overlap_evidence, original_evidence, scorer
                        ),
                    ]

            instance[eva_model] = score_list

            if i % 10 == 0:
                write_json(out_fn, data)

        write_json(out_fn, data)


if __name__ == "__main__":
    for num_facts in [3, 4]:
        minicheck_detection(
            num_facts, ["roberta-large", "deberta-v3-large", "flan-t5-large"]
        )
