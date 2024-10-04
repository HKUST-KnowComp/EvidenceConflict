import re
from tqdm import tqdm
from utils import get_llm_response, load_json, write_json

evidence_conflict_prompt = (
    "Question: {}?\n"
    "Evidence 1: {}\n"
    "Evidence 2: {}\n"
    "Do the two pieces of evidence contain conflicting information on answering the question? (Yes/No)\n"
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


def is_conflict(question, text0, text1, llm_model):
    res = {}
    res[llm_model] = llm_check(question, text0, text1, llm_model)

    return {"conditioned": res}


def llm_detection(dataset, length, reverse, eva_llm_models):

    for eva_llm_model in eva_llm_models:
        in_fn = f"data/{dataset}-{length}-triples-alter_answer.json"
        out_fn = f"data/{dataset}-{length}({eva_llm_model}){reverse}.json"
        # out_fn = f"data/{dataset}-{length}({eva_llm_model})_reverse.json"

        print(f"eva_llm_model is {eva_llm_model}")
        print(f"dataset: {dataset}-{length}{reverse}")

        # if os.path.exists(out_fn):
        #     data = load_json(out_fn)
        # else:
        data = load_json(in_fn)

        if len(data) > 310:
            data = data[:310]

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
                if reverse == "":
                    instance["within_texts"] = [
                        is_conflict(question, x, y, eva_llm_model)
                        for x, y in within_texts
                    ]
                if reverse == "_reverse":
                    instance["within_texts"] = [
                        is_conflict(question, y, x, eva_llm_model)
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
                            is_conflict(question, x, y, eva_llm_model)
                            for x, y in revision_texts
                        ]
                    if reverse == "_reverse":
                        instance["revision_texts"] = [
                            is_conflict(question, y, x, eva_llm_model)
                            for x, y in revision_texts
                        ]
                else:
                    instance["revision_texts"] = None

                if i % 10 == 0:
                    write_json(out_fn, data)

            write_json(out_fn, data)


if __name__ == "__main__":

    eva_llm_models = ["gpt4"]
    for rev in ["", "_reverse"]:
        llm_detection("cwq", "short", rev, eva_llm_models)
