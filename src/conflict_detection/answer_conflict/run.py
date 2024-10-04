from get_data import write_evidence
from sanity_check import sanity_check_dataset
from get_triples import get_triples_set
from pollution import get_pollution
from nli_conflict_checker import nli_detection
from as_conflict_checker import as_detection, minicheck_detection
from llm_conflict_checker import llm_detection
from get_accuracy import get_nli_accuracy, get_llm_accuracy, get_as_accuracy
from utils import load_json, write_json
import os


def prepare_test_dataset(
    dataset,
    length,
    gen_evi_model,
    sanity_check_nli_model,
    sanity_check_llm_model,
    pollute_model,
):
    """
    generate long or short evidence (sentence or paragraph)
    output_fn: f"{dataset}-{length}.json"
    """
    print(f"Processing {dataset}-{length}...")
    write_evidence(dataset, length, gen_evi_model)

    """
    check entailment and consistency
    input_fn: f"{dataset}-{length}.json"
    output_fn: f"{dataset}-{length}-checked.json"
    """
    print("Sanity Checking...")
    sanity_check_dataset(
        dataset, length, sanity_check_nli_model, sanity_check_llm_model
    )

    """
    get triples
    input_fn: f"{dataset}-{length}-checked.json" 
    output_fn: f"{dataset}-{length}-triples.json"
    """
    print("Getting triples...")
    get_triples_set(dataset, length)

    """ 
    pollution 
    input_fn: f"{dataset}-{length}-triples.json"
    output_fn: f"{dataset}-{length}-triples-alter_answer.json"
    """
    print("Generating alternative answers...")
    get_pollution(
        dataset, length, pollute_model, sanity_check_nli_model, sanity_check_llm_model
    )


def evaluate_and_cal_accuracy(
    dataset, length, eva_nli_models, eva_llm_models, eva_as_models, eva_mini_models
):
    """
    evaluate models' performance
    input_fn: f"{dataset}-{length}-triples-alter_answer.json"
    output_fn: f"{dataset}-{length}({llm or nli}).json"
    """
    print("Evaluating NLI Models...")
    nli_detection(dataset, length, eva_nli_models)
    print("Evaluating AlignScore Models...")
    as_detection(dataset, length, eva_as_models)
    minicheck_detection(dataset, length, eva_mini_models)
    print("Evaluating LLM Models...")
    llm_detection(dataset, length, eva_llm_models)

    """
    calculate accuracy rate
    input_fn: f"{dataset}-{length}({llm or nli}).json"
    output_fn: f"results/{dataset}-{length}-accuracy_table.json"
    """
    # print('Getting Accuracy Results...')
    # if os.path.exists(f"results/{dataset}-{length}-accuracy_table.json"):
    #     a_pre = load_json(f"results/{dataset}-{length}-accuracy_table.json")
    #     a_nli = get_nli_accuracy(dataset, length, eva_nli_models, a_pre)
    # else:
    #     a_nli = get_nli_accuracy(dataset, length, eva_nli_models)
    # a = get_nli_accuracy(dataset, length, eva_nli_models)
    # a = get_llm_accuracy(dataset, length, eva_llm_models, a)
    # a = get_as_accuracy(dataset, length, eva_as_models, a)
    # write_json(f"results/{dataset}-{length}-accuracy_table.json", a)


if __name__ == "__main__":

    """args"""
    gen_evi_model = "llama3-70b-instruct"
    sanity_check_nli_model = "microsoft/deberta-v2-xxlarge-mnli"
    sanity_check_llm_model = (
        "llama3-70b-instruct"  # used to generate prediction for nli model to test
    )
    pollute_model = "llama3-70b-instruct"

    eva_nli_models = [
        "microsoft/deberta-xlarge-mnli",
        "microsoft/deberta-v2-xxlarge-mnli",
    ]
    eva_llm_models = [
        "claude-v3-sonnet",
        "claude-v3-haiku",
        "llama3-8b-instruct",
        "llama3-70b-instruct",
        "mixtral-8x7b",
    ]
    # eva_llm_models2 = ["gemini-1.0-pro","gemini-1.5-pro"]
    eva_as_models = ["as-base", "as-large"]
    eva_mini_models = ["roberta-large", "deberta-v3-large", "flan-t5-large"]

    # minicheck_detection("nq_open", "long", eva_mini_models)
    # llm_detection("nq_open", "long", eva_llm_models)

    # evaluate_and_cal_accuracy("cwq", "short", eva_nli_models, eva_llm_models, eva_as_models, eva_mini_models)

    # evaluate_and_cal_accuracy("nq_open", "short", eva_nli_models, eva_llm_models, eva_as_models, eva_mini_models)

    # minicheck_detection("nq_open", "short", eva_mini_models)

    as_detection("nq_open", "long", eva_as_models)
    # llm_detection("cwq", "short", ["mixtral-8x7b"])
    # nli_detection("nq_open", "short", ["microsoft/deberta-v2-xxlarge-mnli"])

    # ------------------------------------------------------

    # step 2: evaluate models' performance
    # evaluate_and_cal_accuracy(dataset, length, eva_nli_models, eva_llm_models)

    # combine cwq's results with cwq2's results
    # for model in eva_nli_models+eva_as_models+eva_llm_models:
    # for model in eva_nli_models:
    #     j1 = load_json(f'data/cwq1-short({model.replace("/","-")}).json')
    #     j2 = load_json(f'data/cwq2-short({model.replace("/","-")}).json')
    #     j = j1+j2
    #     print(model, len(j1),len(j2),len(j))
    #     write_json(f'data/cwq-short({model.replace("/","-")}).json',j)
