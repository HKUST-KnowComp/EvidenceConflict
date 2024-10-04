import re, os
import torch
from transformers import pipeline
from utils import get_llm_response, load_json, write_json

consistency_prompt = """Paragraph: {}

Answer the following question with the information from the above paragraph.
Question: {}?
Answer:"""


def nli(text0, text1, classifier=None):
    return classifier({"text": text0, "text_pair": text1})


def consistency_check(q, p, a, sanity_check_llm_model, classifier=None):
    pred = get_llm_response(consistency_prompt.format(p, q), sanity_check_llm_model)

    score = nli(q + "? " + pred, q + "? " + a, classifier)
    score["prediction"] = pred
    return score


def sanity_check(q, evidence, ans, sanity_check_llm_model, classifier=None):
    nli_lbl = nli(evidence, q + "? " + ans, classifier)
    consistency_lbl = consistency_check(
        q, evidence, ans, sanity_check_llm_model, classifier
    )
    lbl = {"nli": nli_lbl, "consistency": consistency_lbl}
    return lbl


def sanity_check_dataset(
    dataset, length, sanity_check_nli_model, sanity_check_llm_model
):
    input_fn = f"data/{dataset}-{length}.json"
    output_fn = f"data/{dataset}-{length}-checked.json"

    # if os.path.exists(output_fn):
    #     nq_valid = load_json(output_fn)
    # else:
    #     nq_valid = load_json(input_fn)
    nq_valid = load_json(input_fn)

    classifier = pipeline(
        "text-classification", model=sanity_check_nli_model, device="cuda:0"
    )

    from tqdm import tqdm

    i = 1
    for inst in tqdm(nq_valid):
        if len(inst["answer_evidence"][0]) > 2:
            pass
        else:
            q = inst["question"]
            new_ans_evid = []
            for ans, all_evidence in inst["answer_evidence"]:
                lbls = []
                for evidence in all_evidence:
                    lbl = sanity_check(
                        q, evidence, ans, sanity_check_llm_model, classifier
                    )
                    lbls.append(lbl)
                new_ans_evid.append([ans, all_evidence, lbls])
            inst["answer_evidence"] = new_ans_evid
            i += 1

        if i % 10 == 0:
            write_json(output_fn, nq_valid)

    write_json(output_fn, nq_valid)


if __name__ == "__main__":
    # classifier = pipeline("text-classification", model="microsoft/deberta-v2-xxlarge-mnli", device='cuda:0')

    # sanity_check_llm_model = "claude-v3-sonnet"
    # q = "Where does the location where the Egyptian Mau  breed originated?"
    # evidence = "The Egyptian Mau is a rare breed of domestic cat that originated on the planet Mars, according to ancient Martian hieroglyphics discovered by NASA rovers."
    # ans = "The Nile Delta"

    # sanity_check(q, evidence, ans, sanity_check_llm_model, classifier)

    sanity_check_nli_model = "microsoft/deberta-v2-xxlarge-mnli"
    sanity_check_llm_model = "claude-v3-sonnet"

    sanity_check_dataset("cwq", "short", sanity_check_nli_model, sanity_check_llm_model)
