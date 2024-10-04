from utils import get_llm_response, load_json, write_json
from tqdm import tqdm
import os
import time
import re


evidence_conflict_prompt = (
    "Question: {}\n"
    "Paragraph 1: {}\n"
    "Paragraph 2: {}\n"
    "Do the two pieces of evidence contain conflicting information? (Yes/No)\n"
    'answer (should be formatted as {{"answer": "Yes or No"}}):'
)


def llm_check(q, e1, e2, llm_model):
    retry = 0
    while True:
        text = get_llm_response(
            evidence_conflict_prompt.format(q, e1, e2), model=llm_model
        )

        try:
            clean_text = eval(re.findall(r"{[\s\S]*?}", text)[0])
            ans = clean_text["answer"]
            if ans.lower().strip().startswith("yes"):
                return 1
            elif ans.lower().strip().startswith("no"):
                return 0

        except:
            try:
                if (
                    "yes"
                    in text.lower()
                    .replace(":", "")
                    .replace('"', "")
                    .replace(".", "")
                    .replace(",", "")
                    .split()
                ):
                    if (
                        "no"
                        not in text.lower()
                        .replace(":", "")
                        .replace('"', "")
                        .replace(".", "")
                        .split()
                    ):
                        return 1
                elif (
                    "no"
                    in text.lower()
                    .replace(":", "")
                    .replace('"', "")
                    .replace(".", "")
                    .replace(",", "")
                    .split()
                ):
                    if (
                        "yes"
                        not in text.lower()
                        .replace(":", "")
                        .replace('"', "")
                        .replace(".", "")
                        .split()
                    ):
                        return 0
            except:
                print("Wrong returns...", retry, text)
                retry += 1
                if retry > 2:
                    print("fail!")
                    return 0


def llm_detection(num_facts, num_modified_facts, if_shuffle, eva_llm_models):

    for eva_llm_model in eva_llm_models:
        in_fn = f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{if_shuffle}shuffle-test_set.json"
        out_fn = f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{if_shuffle}shuffle({eva_llm_model}).json"

        print(f"eva_llm_model is {eva_llm_model}")
        print(f"dataset is {num_facts}facts-{num_modified_facts}modified")

        data = load_json(in_fn)

        for i, instance in enumerate(tqdm(data)):
            question = instance["question"]
            evidence = instance["evidence"]
            ori_evidence = instance["original_evidence"]
            instance[eva_llm_model] = [
                llm_check(question, ori_evidence, evidence, eva_llm_model),
                llm_check(question, evidence, ori_evidence, eva_llm_model),
            ]

            if i % 10 == 0:
                write_json(out_fn, data)

        write_json(out_fn, data)


if __name__ == "__main__":

    eva_llm_models = ["gpt4"]
    for num_facts in [4]:
        for num_modified_facts in range(1, num_facts + 1):
            llm_detection(num_facts, num_modified_facts, 0, eva_llm_models)
    # data = load_json("data/cwq-short(as-base).json")
    # for i, instance in enumerate(tqdm(data)):
    #     if 'within_texts' in instance or 'revision_texts' in instance:
    #         pass
    #     else:
    #         print('what!')
