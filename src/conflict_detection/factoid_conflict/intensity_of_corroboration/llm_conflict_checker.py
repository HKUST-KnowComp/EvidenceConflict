import re
from get_data import filter_by_nli
from utils import get_llm_response, load_json, write_json
from tqdm import tqdm

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
        print(text)
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


def llm_detection(num_facts, eva_llm_models):

    for eva_llm_model in eva_llm_models:
        in_fn = f"data/strategyqa_shuffled_{num_facts}facts-overlapped-checked.json"
        out_fn = f"test/strategyqa_{num_facts}facts({eva_llm_model}).json"

        print(f"eva_llm_model is {eva_llm_model}")

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
                    llm_check(
                        question, original_evidence, overlap_evidence, eva_llm_model
                    ),
                    llm_check(
                        question, overlap_evidence, original_evidence, eva_llm_model
                    ),
                ]

            # overlap-2:
            res2 = filter_by_nli(nli_scores["overlap-2"])
            if res == True and res2 == True:
                overlap_evidence = evidence_list["overlap-2"]
                score_list["overlap-2"] = [
                    llm_check(
                        question, original_evidence, overlap_evidence, eva_llm_model
                    ),
                    llm_check(
                        question, overlap_evidence, original_evidence, eva_llm_model
                    ),
                ]

            if num_facts == 4:
                # overlap-3:
                res3 = filter_by_nli(nli_scores["overlap-3"])
                if res == True and res3 == True:
                    overlap_evidence = evidence_list["overlap-3"]
                    score_list["overlap-3"] = [
                        llm_check(
                            question, original_evidence, overlap_evidence, eva_llm_model
                        ),
                        llm_check(
                            question, overlap_evidence, original_evidence, eva_llm_model
                        ),
                    ]

            instance[eva_llm_model] = score_list

            if i % 10 == 0:
                write_json(out_fn, data)

        write_json(out_fn, data)


if __name__ == "__main__":
    llm_models = ["gpt4"]
    llm_detection(4, llm_models)
