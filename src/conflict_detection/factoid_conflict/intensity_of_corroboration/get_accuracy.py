import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from utils import load_json, write_json

"""llm"""


def get_llm_results(data, category, eva_model):

    # evidence vs. ori_evidence (has conflict)
    res1 = {}
    preds = []
    for i in range(len(data)):
        try:
            tmp = data[i][eva_model][category][0]
            preds.append(tmp)
        except:
            pass
    true = [1 for _ in range(len(preds))]
    print(f"{category} has {len(preds)} pairs")
    res1[eva_model] = classification_report(true, preds, output_dict=True)["accuracy"]

    # ori_evidence vs. evidence (has conflict)
    res2 = {}
    preds = []
    for i in range(len(data)):
        try:
            tmp = data[i][eva_model][category][1]
            preds.append(tmp)
        except:
            pass
    true = [1 for _ in range(len(preds))]
    # res2[eva_model] = round(classification_report(true, preds, output_dict=True)['accuracy'],4)*100
    res2[eva_model] = classification_report(true, preds, output_dict=True)["accuracy"]

    return res1, res2


def get_llm_accuracy(
    num_facts,
    eva_models,
    accuracy_rates={
        "overlap-1": {"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
        "overlap-2": {"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
        "overlap-3": {"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
    },
):
    for eva_model in eva_models:
        print(f"llm model is {eva_model}")

        in_fn = f"test/strategyqa_{num_facts}facts({eva_model}).json"
        data = load_json(in_fn)

        categories = ["overlap-1", "overlap-2"]
        if num_facts == 4:
            categories.append("overlap-3")

        for category in categories:
            res1, res2 = get_llm_results(data, category, eva_model)

            accuracy_rates[category]["ori_evi vs. evi"].update(res1)
            accuracy_rates[category]["evi vs. ori_evi"].update(res2)

    return accuracy_rates


"""nli"""


def get_nli_results(data, category, eva_model, type_name):

    # evidence vs. ori_evidence (has conflict)
    res1 = {}
    preds = []
    for i in range(len(data)):
        try:
            tmp = data[i][eva_model][category][0][type_name]
            preds.append(tmp)
        except:
            pass

    true = [1 for _ in range(len(preds))]
    res1[f"{eva_model}: {type_name}"] = classification_report(
        true, preds, output_dict=True
    )["accuracy"]

    # ori_evidence vs. evidence (has conflict)
    res2 = {}
    preds_1, preds_2 = [], []
    for i in range(len(data)):
        try:
            tmp = data[i][eva_model][category][1][type_name]
            preds.append(tmp)
        except:
            pass

    true = [1 for _ in range(len(preds))]
    res2[f"{eva_model}: {type_name}"] = classification_report(
        true, preds, output_dict=True
    )["accuracy"]

    return res1, res2


def get_nli_accuracy(
    num_facts,
    eva_models,
    accuracy_rates={
        "overlap-1": {"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
        "overlap-2": {"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
        "overlap-3": {"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
    },
):

    for eva_model in eva_models:
        print(f"nli model is {eva_model}")

        in_fn = f"test/strategyqa_{num_facts}facts({eva_model.replace('microsoft/','')}).json"
        data = load_json(in_fn)

        categories = ["overlap-1", "overlap-2"]
        if num_facts == 4:
            categories.append("overlap-3")

        for category in categories:
            for type_name in ["c>e", "max"]:
                res1, res2 = get_nli_results(
                    data, category, eva_model.replace("microsoft/", ""), type_name
                )

                accuracy_rates[category]["ori_evi vs. evi"].update(res1)
                accuracy_rates[category]["evi vs. ori_evi"].update(res2)

    return accuracy_rates


"""align score"""


def get_as_results(data, category, eva_model):

    # evidence vs. ori_evidence (has conflict)
    res1 = {}
    preds = []
    for i in range(len(data)):
        try:
            tmp = data[i][eva_model][category][0]
            preds.append(tmp)
        except:
            pass
    true = [1 for _ in range(len(preds))]
    res1[eva_model] = classification_report(true, preds, output_dict=True)["accuracy"]

    # ori_evidence vs. evidence (has conflict)
    res2 = {}
    preds = []
    for i in range(len(data)):
        try:
            tmp = data[i][eva_model][category][1]
            preds.append(tmp)
        except:
            pass
    true = [1 for _ in range(len(preds))]
    res2[eva_model] = classification_report(true, preds, output_dict=True)["accuracy"]

    return res1, res2


def get_as_accuracy(
    num_facts,
    eva_models,
    accuracy_rates={
        "overlap-1": {"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
        "overlap-2": {"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
        "overlap-3": {"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
    },
):

    for eva_model in eva_models:
        print(f"as model is {eva_model}")

        in_fn = f"test/strategyqa_{num_facts}facts({eva_model}).json"
        data = load_json(in_fn)

        categories = ["overlap-1", "overlap-2"]
        if num_facts == 4:
            categories.append("overlap-3")

        for category in categories:
            res1, res2 = get_as_results(data, category, eva_model)

            accuracy_rates[category]["ori_evi vs. evi"].update(res1)
            accuracy_rates[category]["evi vs. ori_evi"].update(res2)

    return accuracy_rates


"""mini check"""


def get_mini_results(data, category, eva_model):

    # evidence vs. ori_evidence (has conflict)
    res1 = {}
    preds = []
    for i in range(len(data)):
        try:
            tmp = data[i][eva_model][category][0]
            preds.append(tmp)
        except:
            pass
    true = [1 for _ in range(len(preds))]
    res1[eva_model] = classification_report(true, preds, output_dict=True)["accuracy"]

    # ori_evidence vs. evidence (has conflict)
    res2 = {}
    preds = []
    for i in range(len(data)):
        try:
            tmp = data[i][eva_model][category][1]
            preds.append(tmp)
        except:
            pass
    true = [1 for _ in range(len(preds))]
    res2[eva_model] = classification_report(true, preds, output_dict=True)["accuracy"]

    return res1, res2


def get_mini_accuracy(
    num_facts,
    eva_models,
    accuracy_rates={
        "overlap-1": {"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
        "overlap-2": {"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
        "overlap-3": {"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
    },
):

    for eva_model in eva_models:
        print(f"as model is {eva_model}")

        in_fn = f"test/strategyqa_{num_facts}facts({eva_model}).json"
        data = load_json(in_fn)

        categories = ["overlap-1", "overlap-2"]
        if num_facts == 4:
            categories.append("overlap-3")

        for category in categories:
            res1, res2 = get_mini_results(data, category, eva_model)

            accuracy_rates[category]["ori_evi vs. evi"].update(res1)
            accuracy_rates[category]["evi vs. ori_evi"].update(res2)

    return accuracy_rates


"""aggregate to table"""


def get_accuracy_table(num_facts, models):
    result_df = pd.DataFrame()
    result_df["model"] = models

    # for each modified facts
    for num_modified_facts in range(1, num_facts):
        modified_df = pd.DataFrame()
        modified_df["model"] = models

        res_list = []
        data = load_json(f"result/{num_facts}facts-singlepairs.json")

        for model in models:
            tmp = np.mean(
                [
                    data[f"overlap-{num_modified_facts}"]["evi vs. ori_evi"][model],
                    data[f"overlap-{num_modified_facts}"]["ori_evi vs. evi"][model],
                ]
            )
            tmp = round(tmp * 100, 2)
            res_list.append(tmp)

        modified_df[f"({num_modified_facts}overlap)"] = res_list
        result_df = result_df.merge(modified_df, on=["model"])

    model_map = {
        "mixtral-8x7b": "Mixtral 8x7B",
        "llama3-8b-instruct": "Llama-3 8B Inst.",
        "llama3-70b-instruct": "Llama-3 70B Inst.",
        "claude-v3-haiku": "Claude 3 Haiku",
        "claude-v3-sonnet": "Claude 3 Sonnet",
        "as-base": "AlignScore-base",
        "as-large": "AlignScore-large",
        "roberta-large": "MiniCheck-R",
        "deberta-v3-large": "MiniCheck-D",
        "flan-t5-large": "MiniCheck-FT5",
        "deberta-xlarge-mnli: max": "NLI-xlarge (Max)",
        "deberta-xlarge-mnli: c>e": "NLI-xlarge (C>E)",
        "deberta-v2-xxlarge-mnli: max": "NLI-xxlarge (Max)",
        "deberta-v2-xxlarge-mnli: c>e": "NLI-xxlarge (C>E)",
        "chatgpt": "ChatGPT",
        "gpt4": "GPT4",
    }

    all_models = [
        "Mixtral 8x7B",
        "Llama-3 8B Inst.",
        "Llama-3 70B Inst.",
        "Claude 3 Haiku",
        "Claude 3 Sonnet",
        "ChatGPT",
        "GPT4",
        "AlignScore-base",
        "AlignScore-large",
        "MiniCheck-R",
        "MiniCheck-D",
        "MiniCheck-FT5",
        "NLI-xlarge (Max)",
        "NLI-xlarge (C>E)",
        "NLI-xxlarge (Max)",
        "NLI-xxlarge (C>E)",
    ]

    result_df["model"] = result_df["model"].apply(lambda x: model_map[x])
    result_df["model"] = pd.Categorical(
        result_df["model"], categories=all_models, ordered=True
    )
    result_df = result_df.groupby(["model"]).first()

    return result_df


def plot_accuracy_intensity(num_facts, plot_models, order="evi vs. ori_evi"):
    import matplotlib.pyplot as plt

    model_accuracy = {}

    for model in plot_models:
        accs = []

        data = load_json(f"result/{num_facts}facts-overlappairs.json")
        for num_modified_facts in range(1, num_facts):
            accs.append(data[f"overlap-{num_modified_facts}"][order][model])

        model_accuracy[model] = accs

    plt.figure(figsize=(10, 10))
    plt.title(f"Models' accuracy performance for on ({num_facts}facts):")
    plt.xlabel("Number of overlapped facts")
    plt.ylabel("Accuracy Score")
    plt.ylim(0, 1)
    plt.xticks(range(1, num_facts + 1))

    for model in plot_models:
        plt.plot(range(1, num_facts), model_accuracy[model], "-o", label=model)

    plt.legend()
    plt.show()
    plt.savefig(f"result/figures/acc-{num_facts}facts-overlappairs.png")


if __name__ == "__main__":

    # a = load_json('result/4facts-singlepairs.json')
    # a = get_llm_accuracy(4, ['gpt4'], a)
    # write_json('result/4facts-singlepairs.json', a)
    # print(a)

    # a = get_llm_accuracy(num_facts, ['claude-v3-sonnet'])
    # a = get_as_accuracy(num_facts, ['as-base'], a)
    # a = get_nli_accuracy(num_facts,['microsoft/deberta-large-mnli'], a)
    # print(a)

    # eva_llm_models = ["claude-v3-sonnet", "claude-v3-haiku","llama3-8b-instruct","llama3-70b-instruct","mixtral-8x7b"]
    # eva_as_models = ["as-base", "as-large"]
    # eva_nli_models = ["microsoft/deberta-xlarge-mnli", "microsoft/deberta-v2-xxlarge-mnli"]
    # eva_mini_models = ['roberta-large', 'deberta-v3-large', 'flan-t5-large']

    # for num_facts in [3,4]:

    #     a = {"overlap-1":{"evi vs. ori_evi":{}, "ori_evi vs. evi":{}},"overlap-2":{"evi vs. ori_evi":{}, "ori_evi vs. evi":{}},"overlap-3":{"evi vs. ori_evi":{}, "ori_evi vs. evi":{}}}
    #     a = get_as_accuracy(num_facts, eva_as_models, a)
    #     a = get_mini_accuracy(num_facts, eva_mini_models, a)
    #     a = get_llm_accuracy(num_facts, eva_llm_models, a)
    #     a = get_nli_accuracy(num_facts, eva_nli_models, a)

    #     write_json(f"result/{num_facts}facts-singlepairs.json", a)

    models = [
        "chatgpt",
        "gpt4",
        "deberta-xlarge-mnli: c>e",
        "deberta-xlarge-mnli: max",
        "deberta-v2-xxlarge-mnli: c>e",
        "deberta-v2-xxlarge-mnli: max",
        "as-base",
        "as-large",
        "claude-v3-sonnet",
        "claude-v3-haiku",
        "llama3-8b-instruct",
        "llama3-70b-instruct",
        "mixtral-8x7b",
        "flan-t5-large",
        "roberta-large",
        "deberta-v3-large",
    ]

    res = get_accuracy_table(4, models)

    # res1 = get_accuracy_table(3, models)
    # res2 = get_accuracy_table(4, models)
    # res = pd.concat([res1,res2])
    res.to_csv("result/accuracy_overlap_4facts.csv")
