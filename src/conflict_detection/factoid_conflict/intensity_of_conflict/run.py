import pandas as pd

from get_data import get_modified_facts, consistency_check, get_mixed_facts_evidence
from get_test_set import get_set_for_test
from llm_conflict_checker import llm_detection
from nli_conflict_checker import nli_detection
from as_conflict_checker import as_detection
from get_accuracy import (
    get_as_accuracy,
    get_nli_accuracy,
    get_llm_accuracy,
    plot_for_intensity,
    get_result_table,
    get_accuracy_table,
    plot_accuracy_intensity,
)
from utils import load_json, write_json


"""generate datasets"""
gen_model = "llama3-70b-instruct"
# for num_facts in [3]:
#     # print(f'Modifying facts for {num_facts}facts...')
#     # get_modified_facts(num_facts, gen_model)

#     # print(f'Generating evidence for <{num_facts}facts-0modified-0shuffle>...')
#     # get_mixed_facts_evidence(num_facts, 0, gen_model, 0)

#     # for if_shuffle in [0, 1]:
#     #     for num_modified_facts in range(1, num_facts+1):
#     #         print(f'Generating evidence for <{num_facts}facts-{num_modified_facts}modified-{if_shuffle}shuffle>...')
#     #         get_mixed_facts_evidence(num_facts, num_modified_facts, gen_model, if_shuffle)


#     '''check consistency'''
#     check_nli_model = "microsoft/deberta-v2-xxlarge-mnli"
#     # print(f'Checking consistency for <{num_facts}facts-0modified-0shuffle>...')
#     # consistency_check(num_facts, 0, 0, check_nli_model)

#     # for if_shuffle in [0, 1]:
#     #     for num_modified_facts in range(1, num_facts+1):
#     #         print(f'Checking consistency for <{num_facts}facts-{num_modified_facts}modified-{if_shuffle}shuffle>...')
#     #         consistency_check(num_facts, num_modified_facts, if_shuffle, check_nli_model)
#     print(f'Checking consistency for <{num_facts}facts-3modified-0shuffle>...')
#     consistency_check(num_facts, 3, 0, check_nli_model)
#     print(f'Checking consistency for <{num_facts}facts-3modified-1shuffle>...')
#     consistency_check(num_facts, 3, 1, check_nli_model)
#     print(f'Checking consistency for <{num_facts}facts-2modified-1shuffle>...')
#     consistency_check(num_facts, 2, 1, check_nli_model)
#     print(f'Checking consistency for <{num_facts}facts-1modified-1shuffle>...')
#     consistency_check(num_facts, 1, 1, check_nli_model)


#     '''get test set'''
#     for if_shuffle in [0,1]:
#         for num_modified_facts in range(1, num_facts+1):
#             print(f'Getting test set for <{num_facts}facts-{num_modified_facts}modified-{if_shuffle}shuffle>...')
#             get_set_for_test(num_facts, num_modified_facts, if_shuffle)


for num_facts in [4]:
    """llm detection"""
    eva_llm_models = [
        "claude-v3-sonnet",
        "claude-v3-haiku",
        "llama3-8b-instruct",
        "llama3-70b-instruct",
        "mixtral-8x7b",
    ]
    for num_modified_facts in range(1, num_facts + 1):
        for if_shuffle in [0, 1]:
            print(
                f"LLM detection for <{num_facts}facts-{num_modified_facts}modified-{if_shuffle}shuffle>..."
            )
            llm_detection(num_facts, num_modified_facts, if_shuffle, eva_llm_models)

for num_facts in [3, 4]:
    """nli detection"""
    eva_nli_models = [
        "microsoft/deberta-xlarge-mnli",
        "microsoft/deberta-v2-xxlarge-mnli",
    ]
    for num_modified_facts in range(1, num_facts + 1):
        for if_shuffle in [0, 1]:
            print(
                f"NLI detection for <{num_facts}facts-{num_modified_facts}modified-{if_shuffle}shuffle>..."
            )
            nli_detection(num_facts, num_modified_facts, if_shuffle, eva_nli_models)

    """as detection"""
    eva_as_models = ["as-base", "as-large"]
    for num_modified_facts in range(1, num_facts + 1):
        for if_shuffle in [0, 1]:
            print(
                f"AS detection for <{num_facts}facts-{num_modified_facts}modified-{if_shuffle}shuffle>..."
            )
            as_detection(num_facts, num_modified_facts, if_shuffle, eva_as_models)

    """get accuracy"""
    for num_modified_facts in range(1, num_facts + 1):
        for is_shuffle in [0, 1]:
            a = get_as_accuracy(
                num_facts, num_modified_facts, is_shuffle, eva_as_models
            )
            a = get_llm_accuracy(
                num_facts, num_modified_facts, is_shuffle, eva_llm_models, a
            )
            a = get_nli_accuracy(
                num_facts, num_modified_facts, is_shuffle, eva_nli_models, a
            )

            write_json(
                f"result/s-{num_facts}facts-{num_modified_facts}modified-{is_shuffle}shuffle-result.json",
                a,
            )

    models = [
        "claude-v3-sonnet",
        "claude-v3-haiku",
        "llama3-8b-instruct",
        "llama3-70b-instruct",
        "mixtral-8x7b",
        "as-base",
        "as-large",
        "deberta-xlarge-mnli-conditioned-c>e",
        "deberta-xlarge-mnli-conditioned-max",
        "deberta-v2-xxlarge-mnli-conditioned-c>e",
        "deberta-v2-xxlarge-mnli-conditioned-max",
    ]
    for is_shuffle in [0, 1]:
        plot_accuracy_intensity(num_facts, is_shuffle, models)
        res = get_accuracy_table(num_facts, is_shuffle, models)
        res.to_csv(f"result/tables/acc-{num_facts}facts-{is_shuffle}shuffle.csv")
