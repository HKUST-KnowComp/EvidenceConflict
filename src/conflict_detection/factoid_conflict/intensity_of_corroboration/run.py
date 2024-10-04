import pandas as pd

from get_data import (
    shuffle_facts,
    get_overlapped_facts_evidence,
    consistency_check,
    get_set_for_test,
)
from llm_conflict_checker import llm_detection
from nli_conflict_checker import nli_detection
from as_conflict_checker import as_detection
from get_accuracy import get_llm_accuracy, get_nli_accuracy, get_as_accuracy
from utils import load_json, write_json

"""args"""
gen_evi_model = "llama3-70b-instruct"
check_nli_model = "microsoft/deberta-v2-xxlarge-mnli"
eva_llm_models = [
    "claude-v3-sonnet",
    "claude-v3-haiku",
    "llama3-8b-instruct",
    "llama3-70b-instruct",
    "mixtral-8x7b",
]
eva_nli_models = ["microsoft/deberta-xlarge-mnli", "microsoft/deberta-v2-xxlarge-mnli"]
eva_as_models = ["as-base", "as-large"]


for num_facts in [3]:
    # print(f'Getting test set for {num_facts}facts...')

    # '''shuffle facts'''
    # shuffle_facts(num_facts)

    # '''generate evidence'''
    # get_overlapped_facts_evidence(num_facts, gen_evi_model)

    # '''consistency check'''
    # consistency_check(num_facts, check_nli_model)

    # '''get test set'''
    # get_set_for_test(num_facts)

    """llm detection"""
    llm_detection(num_facts, ["mixtral-8x7b"])

    # '''nli detection'''
    # nli_detection(num_facts, ["microsoft/deberta-v2-xxlarge-mnli"])

    # '''as detection'''
    # as_detection(num_facts, eva_as_models)

    # '''get accuracy'''
    # a = get_llm_accuracy(num_facts,eva_llm_models)
    # a = get_as_accuracy(num_facts,eva_as_models, a)
    # a = get_nli_accuracy(num_facts,eva_nli_models, a)
    # write_json(f'result/strategyqa_{num_facts}facts-overlapped-accuracy.json',a)
