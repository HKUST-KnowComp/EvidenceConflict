import os
import re
from tqdm import tqdm
from datasets import load_dataset
from utils import get_llm_response, write_json, load_json

parametric_answer_prompt = """Give a short answer to the question.

Question: {}?
Answer:"""


def get_parametric_answer(q, gen_evi_model):
    res = get_llm_response(parametric_answer_prompt.format(q), gen_evi_model)
    return res.strip().split("\n")


possible_answer_prompt = (
    "List THREE different short answers to the question. The answers do not have to be true.\n"
    "Question: {}?\n"
    'Answer (should be formatted as {{"1": "TEXT-1", "2": "TEXT-2", "3": "TEXT-3"}}):'
)


def get_possible_answers(q, gen_evi_model):
    res = get_llm_response(possible_answer_prompt.format(q), gen_evi_model)
    clean_res = eval(re.findall(r"{[\s\S]*?}", res)[0])
    res = [clean_res["1"], clean_res["2"], clean_res["3"]]
    return res


evidence_prompt = (
    "Give me TWO different short paragraphs that independently support the given answer (try to simulate the format of web search results).\n"
    "Question: {}?\n"
    "Answer: {}\n"
    'Paragraphs (should be formatted as {{"1": "TEXT-1", "2": "TEXT-2"}}):'
)

evidence_short_prompt = (
    "Give me TWO different short sentences that independently support the given answer (try to simulate the format of web search results).\n"
    "Question: {}?\n"
    "Answer: {}\n"
    'Sentences (should be formatted as {{"1": "TEXT-1", "2": "TEXT-2"}}):'
)


def get_evidence(q, a, gen_evi_model):
    res = get_llm_response(evidence_prompt.format(q, a), gen_evi_model)
    clean_res = eval(re.findall(r"{[\s\S]*?}", res)[0])
    res = [clean_res["1"], clean_res["2"]]

    return res


def get_short_evidence(q, a, gen_evi_model):
    res = get_llm_response(evidence_short_prompt.format(q, a), gen_evi_model)
    clean_res = eval(re.findall(r"{[\s\S]*?}", res)[0])
    res = [clean_res["1"], clean_res["2"]]

    return res


def write_evidence(dataset, length, gen_evi_model):

    to_fn = f"data/{dataset}-{length}.json"

    if length == "short":
        evidence_func = get_short_evidence
        number = 500
    elif length == "long":
        evidence_func = get_evidence
        number = 500
    else:
        print("Wrong length")

    if os.path.exists(to_fn):
        collected_data = load_json(to_fn)
    else:
        collected_data = []

    if dataset == "nq_open":
        all_dataset = load_dataset("nq_open")["validation"]
    elif dataset == "cwq":
        all_dataset = load_dataset(
            "drt/complex_web_questions", "complex_web_questions"
        )["validation"]
    else:
        print("Wrong dataset name1")

    from random import sample, seed

    seed(2023)
    dataset_ids = sample(range(len(all_dataset)), number)
    id_num = 0
    data_set = [all_dataset[id_] for id_ in dataset_ids]

    for i in tqdm(range(len(collected_data), number)):
        retr_ct = 0
        while retr_ct < 3:

            try:
                it = data_set[i]
                q = it["question"]
                if dataset == "nq_open":
                    gt_a = it["answer"]
                elif dataset == "cwq":
                    gt_a = it["answers"]["answer"]
                else:
                    print("Wrong dataset name2")

                answers = get_possible_answers(q, gen_evi_model)
                this = {
                    "question": q,
                    "groundtruth_answer": gt_a,
                    "possible_answer": answers,
                    "answer_evidence": [],
                }

                tmp = answers + gt_a
                answers = list(set(tmp))
                for a in answers:
                    e = evidence_func(q, a, gen_evi_model)
                    this["answer_evidence"].append((a, e))
                collected_data.append(this)

                id_num += 1

                break

            except Exception as e:
                retr_ct += 1
                print(e)
                print("Retry: ", retr_ct)

        if i % 10 == 0:
            print("Saving...")
            write_json(to_fn, collected_data)

    write_json(to_fn, collected_data)


if __name__ == "__main__":

    gen_evi_model = "claude-v3-sonnet"

    write_evidence("cwq", "short", gen_evi_model)
