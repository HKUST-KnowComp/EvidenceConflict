# polluting the triple data with various strategies

import re
from sanity_check import sanity_check
from transformers import pipeline
from tqdm import tqdm
from utils import get_llm_response, load_json, write_json

revise_answer_prompt = (
    "Given the following passage, modify as few details as possible to make it support the given answer to the question.\n"
    "\n"
    "Question: {}?\n"
    "Passage: {}\n"
    "Answer: {}\n"
    'Modified passage (should be formatted as {{"Modified_passage": "TEXT"}}):'
)


def revise_answer(
    question, evidence, answer, pollute_model, sanity_check_llm_model, classifier=None
):
    good_quality = False
    retr_ct = 0
    while retr_ct < 3:
        revision = get_llm_response(
            revise_answer_prompt.format(question, evidence, answer), pollute_model
        )

        try:
            clean_rev = eval(re.findall(r"{[\s\S]*?}", revision)[0])
            revision = clean_rev["Modified_passage"]
            lbl = sanity_check(
                question, revision, answer, sanity_check_llm_model, classifier
            )
            good_quality = (
                lbl["nli"]["label"] == "ENTAILMENT"
                and lbl["consistency"]["label"] == "ENTAILMENT"
            )

            break

        except Exception as e:
            retr_ct += 1
            print("Retry: ", retr_ct)

    if good_quality:
        return revision, lbl


def get_pollution(
    dataset, length, pollute_model, sanity_check_nli_model, sanity_check_llm_model
):
    in_fn = f"data/{dataset}-{length}-triples.json"
    out_fn = f"data/{dataset}-{length}-triples-alter_answer.json"

    classifier = pipeline(
        "text-classification", model=sanity_check_nli_model
    )  # , device='cuda:0'

    triples = load_json(in_fn)

    for i, triple in enumerate(tqdm(triples)):
        q = triple["question"]
        evidence = triple["texts"][0]
        target_answer = triple["answers"][2]

        retry = 0
        while retry < 2:
            revision = revise_answer(
                q,
                evidence,
                target_answer,
                pollute_model,
                sanity_check_llm_model,
                classifier,
            )
            if revision is not None:
                break
            retry += 1
            print("Retrying generation as the quality is bad.", retry)

        triple["revision"] = revision[0] if revision else None

        if i % 10 == 0:
            write_json(out_fn, triples)

    write_json(out_fn, triples)


if __name__ == "__main__":
    dataset = "cwq"
    length = "short"  # ['short', 'long']

    gen_evi_model = "claude-v3-sonnet"
    sanity_check_nli_model = "microsoft/deberta-v2-xxlarge-mnli"
    sanity_check_llm_model = (
        "claude-v3-sonnet"  # used to generate prediction for nli model to test
    )
    pollute_model = "claude-v3-sonnet"

    get_pollution(
        dataset, length, pollute_model, sanity_check_nli_model, sanity_check_llm_model
    )
