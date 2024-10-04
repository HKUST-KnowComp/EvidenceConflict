from sklearn.metrics import classification_report
from utils import load_json, write_json

# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

"""llm"""


def get_llm_results(data, eva_model):

    # evidence vs. ori_evidence (has conflict)
    res1 = {}
    preds = [i[eva_model][0] for i in data if i[eva_model]]
    true = [1 for _ in range(len(preds))]
    # res1[eva_model] = round(classification_report(true, preds, output_dict=True)['accuracy'],4)*100
    res1[eva_model] = classification_report(true, preds, output_dict=True)["accuracy"]

    # ori_evidence vs. evidence (has conflict)
    res2 = {}
    preds = [i[eva_model][1] for i in data if i[eva_model]]
    true = [1 for _ in range(len(preds))]
    # res2[eva_model] = round(classification_report(true, preds, output_dict=True)['accuracy'],4)*100
    res2[eva_model] = classification_report(true, preds, output_dict=True)["accuracy"]

    return res1, res2


def get_llm_accuracy(
    num_facts,
    num_modified_facts,
    is_shuffle,
    modify,
    eva_models,
    accuracy_rates={"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
):
    for eva_model in eva_models:
        print(f"llm model is {eva_model}")

        data = load_json(
            f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{is_shuffle}shuffle({eva_model}).json"
        )
        if len(data) > 300:
            data = data[:300]

        res1, res2 = get_llm_results(data, eva_model)
        accuracy_rates[modify]["evi vs. ori_evi"].update(res1)
        accuracy_rates[modify]["ori_evi vs. evi"].update(res2)

    return accuracy_rates


"""nli"""


def aggregate_pred_nli(all_preds, type_name):
    agg = {"conditioned": {type_name: []}}

    for p in all_preds:
        for k in agg:
            if k == "conditioned":
                for t in agg[k]:
                    if t == type_name:
                        agg[k][t].append(p[k][t])
    return agg


def get_nli_results(data, eva_model, type_name):

    # evidence vs. ori_evidence (has conflict)
    res1 = {}
    preds = [i[eva_model][0] for i in data if i[eva_model]]
    pred = aggregate_pred_nli(preds, type_name)
    true = [1 for _ in range(len(pred["conditioned"][type_name]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            # res1[f"{eva_model}-{k}-{t}"] = round(classification_report(true, p, output_dict=True)['accuracy'],4)*100
            res1[f"{eva_model}: {type_name}"] = classification_report(
                true, p, output_dict=True
            )["accuracy"]

        # res1[f"{eva_model}-{k}-average"] = round((res1[f"{eva_model}-{k}-c>e"]+res1[f"{eva_model}-{k}-max"])/2,2)

    # ori_evidence vs. evidence (has conflict)
    res2 = {}
    preds = [i[eva_model][1] for i in data if i[eva_model]]
    pred = aggregate_pred_nli(preds, type_name)
    true = [1 for _ in range(len(pred["conditioned"][type_name]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            # res2[f"{eva_model}-{k}-{t}"] = round(classification_report(true, p, output_dict=True)['accuracy'],4)*100
            res2[f"{eva_model}: {type_name}"] = classification_report(
                true, p, output_dict=True
            )["accuracy"]

        # res2[f"{eva_model}-{k}-average"] = round((res2[f"{eva_model}-{k}-c>e"]+res2[f"{eva_model}-{k}-max"])/2,2)

    return res1, res2


def get_nli_accuracy(
    num_facts,
    num_modified_facts,
    is_shuffle,
    eva_models,
    accuracy_rates={"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
):

    for eva_model in eva_models:
        print(f"nli model is {eva_model}")

        data = load_json(
            f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{is_shuffle}shuffle({eva_model.replace('microsoft/','')}).json"
        )
        if len(data) > 300:
            data = data[:300]

        for type_name in ["c>e", "max"]:
            res1, res2 = get_nli_results(
                data, eva_model.replace("microsoft/", ""), type_name
            )
            accuracy_rates["evi vs. ori_evi"].update(res1)
            accuracy_rates["ori_evi vs. evi"].update(res2)

    return accuracy_rates


"""align score"""


def aggregate_pred_as(all_preds, eva_model):
    res = {}
    res[eva_model] = []
    agg = {"conditioned": res}

    for p in all_preds:
        for k in agg:
            if k == "conditioned":
                for t in agg[k]:
                    agg[k][t].append(p[k][t])
    return agg


def get_as_results(data, eva_model):

    # evidence vs. ori_evidence (has conflict)
    res1 = {}
    preds = [i[eva_model][0]["conditioned"][eva_model] for i in data if i[eva_model]]
    true = [1 for _ in range(len(preds))]
    res1[eva_model] = classification_report(true, preds, output_dict=True)["accuracy"]

    # ori_evidence vs. evidence (has conflict)
    res2 = {}
    preds = [i[eva_model][1]["conditioned"][eva_model] for i in data if i[eva_model]]
    true = [1 for _ in range(len(preds))]
    res2[eva_model] = classification_report(true, preds, output_dict=True)["accuracy"]

    return res1, res2


def get_as_accuracy(
    num_facts,
    num_modified_facts,
    is_shuffle,
    eva_models,
    accuracy_rates={"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
):
    for eva_model in eva_models:
        print(f"as model is {eva_model}")

        data = load_json(
            f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{is_shuffle}shuffle({eva_model}).json"
        )
        if len(data) > 300:
            data = data[:300]

        res1, res2 = get_as_results(data, eva_model)
        accuracy_rates["evi vs. ori_evi"].update(res1)
        accuracy_rates["ori_evi vs. evi"].update(res2)

    return accuracy_rates


"""minicheck"""


def aggregate_pred_mini(all_preds, eva_model):
    res = {}
    res[eva_model] = []
    agg = {"conditioned": res}

    for p in all_preds:
        for k in agg:
            if k == "conditioned":
                for t in agg[k]:
                    agg[k][t].append(p[k][t])
    return agg


def get_mini_results(data, eva_model):

    # evidence vs. ori_evidence (has conflict)
    res1 = {}
    preds = [i[eva_model][0]["conditioned"][eva_model] for i in data if i[eva_model]]
    true = [1 for _ in range(len(preds))]
    res1[eva_model] = classification_report(true, preds, output_dict=True)["accuracy"]

    # ori_evidence vs. evidence (has conflict)
    res2 = {}
    preds = [i[eva_model][1]["conditioned"][eva_model] for i in data if i[eva_model]]
    true = [1 for _ in range(len(preds))]
    res2[eva_model] = classification_report(true, preds, output_dict=True)["accuracy"]

    return res1, res2


def get_mini_accuracy(
    num_facts,
    num_modified_facts,
    is_shuffle,
    eva_models,
    accuracy_rates={"evi vs. ori_evi": {}, "ori_evi vs. evi": {}},
):
    for eva_model in eva_models:
        print(f"as model is {eva_model}")

        data = load_json(
            f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{is_shuffle}shuffle({eva_model}).json"
        )
        if len(data) > 300:
            data = data[:300]

        res1, res2 = get_as_results(data, eva_model)
        accuracy_rates["evi vs. ori_evi"].update(res1)
        accuracy_rates["ori_evi vs. evi"].update(res2)

    return accuracy_rates


"""aggregate to table"""


def get_result_table(num_facts, is_shuffle, models):
    result_df = pd.DataFrame()
    result_df["model"] = models

    for order in ["evi vs. ori_evi", "ori_evi vs. evi"]:
        order_df = pd.DataFrame()
        order_df["model"] = models

        for measure in ["precision", "recall", "f1-score"]:
            measure_df = pd.DataFrame()
            measure_df["model"] = models

            # for each modified facts
            for num_modified_facts in range(1, num_facts + 1):
                modified_df = pd.DataFrame()
                modified_df["model"] = models

                res_list = []

                data = load_json(
                    f"result/s-{num_facts}facts-{num_modified_facts}modified-{is_shuffle}shuffle-result.json"
                )
                for model in models:
                    res_list.append(data[order][model]["macro avg"][measure])
                modified_df[f"({num_modified_facts}modified)-({measure})-({order})"] = (
                    res_list
                )

                measure_df = measure_df.merge(modified_df, on=["model"])

            order_df = order_df.merge(measure_df, on=["model"])

        result_df = result_df.merge(order_df, on=["model"])

    result_df.set_index("model")

    return result_df


def get_accuracy_table_flatten(models):
    result_df = pd.DataFrame()

    for num_facts in [3, 4]:
        facts_df = pd.DataFrame()

        for is_shuffle in [0, 1]:
            shuffle_df = pd.DataFrame()

            data = load_json(
                f"result/{num_facts}facts_{is_shuffle}shuffle_singlepairs.json"
            )

            # for each modified facts
            for num_modified_facts in range(1, num_facts + 1):
                modified_df = pd.DataFrame()

                res_list = []

                for model in models:
                    acctmp = 0
                    for order in ["evi vs. ori_evi", "ori_evi vs. evi"]:
                        acctmp += data[f"{num_modified_facts}modified"][order][model]

                    res_list.append(round(acctmp * 100 / 2, 2))

                modified_df[f"accuracy"] = res_list
                modified_df["modify"] = f"{num_modified_facts}modified"
                modified_df["model"] = models

                shuffle_df = pd.concat([shuffle_df, modified_df])

            shuffle_df["shuffle"] = "shuffled" if is_shuffle == 1 else "not shuffled"
            facts_df = pd.concat([facts_df, shuffle_df])

        facts_df["facts"] = f"{num_facts}facts"
        result_df = pd.concat([result_df, facts_df])

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
    }

    result_df["model"] = result_df["model"].apply(lambda x: model_map[x])
    print(result_df)
    return result_df


def get_accuracy_table(models):
    result_df = pd.DataFrame()

    for num_facts in [4]:
        facts_df = pd.DataFrame()

        for is_shuffle in [0]:
            shuffle_df = pd.DataFrame()
            shuffle_df["model"] = models

            data = load_json(
                f"result/{num_facts}facts_{is_shuffle}shuffle_singlepairs.json"
            )

            # for each modified facts
            for num_modified_facts in range(1, num_facts + 1):
                modified_df = pd.DataFrame()

                res_list = []

                for model in models:
                    acctmp = 0
                    for order in ["evi vs. ori_evi", "ori_evi vs. evi"]:
                        acctmp += data[f"{num_modified_facts}modified"][order][model]

                    res_list.append(round(acctmp * 100 / 2, 2))

                modified_df[f"{num_modified_facts}modified"] = res_list
                modified_df["model"] = models

                shuffle_df = shuffle_df.merge(modified_df, on=["model"])

            shuffle_df["shuffle"] = "shuffled" if is_shuffle == 1 else "not shuffled"
            facts_df = pd.concat([facts_df, shuffle_df])

        facts_df["facts"] = f"{num_facts}facts"
        result_df = pd.concat([result_df, facts_df])

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

    result_df = result_df.groupby(["shuffle", "facts", "model"]).first()

    return result_df


def plot_for_intensity(num_facts, is_shuffle, plot_models):
    import matplotlib.pyplot as plt

    for model in plot_models:

        precisions, recalls, f1s = [], [], []

        for num_modified_facts in range(1, num_facts + 1):
            data = load_json(
                f"result/s-{num_facts}facts-{num_modified_facts}modified-{is_shuffle}shuffle-result.json"
            )
            precisions.append(data["evi vs. ori_evi"][model]["macro avg"]["precision"])
            recalls.append(data["evi vs. ori_evi"][model]["macro avg"]["recall"])
            f1s.append(data["evi vs. ori_evi"][model]["macro avg"]["f1-score"])

        plt.figure(figsize=(10, 10))
        plt.title(
            f"Model Performance for <{model}> on {num_facts}facts-{is_shuffle}shuffle:"
        )
        plt.xlabel("Number of modified facts")
        plt.ylabel("Score")
        plt.ylim(0, 1)

        plt.xticks(range(1, num_facts + 1))
        plt.plot(
            range(1, num_facts + 1), precisions, "-o", label="precision", c="orange"
        )
        plt.plot(range(1, num_facts + 1), recalls, "-x", label="recall", c="blue")
        plt.plot(range(1, num_facts + 1), f1s, "-^", label="f1-score", c="green")

        plt.legend()
        plt.show()
        plt.savefig(
            f"result/figures/<{model}>-{num_facts}facts-{is_shuffle}shuffle.png"
        )


def plot_accuracy_intensity(
    num_facts, is_shuffle, plot_models, order="evi vs. ori_evi"
):
    import matplotlib.pyplot as plt

    model_accuracy = {}

    for model in plot_models:
        accs = []
        for num_modified_facts in range(1, num_facts + 1):
            data = load_json(
                f"result/s-{num_facts}facts-{num_modified_facts}modified-{is_shuffle}shuffle-result.json"
            )
            accs.append(data[order][model]["accuracy"])

        model_accuracy[model] = accs

    plt.figure(figsize=(10, 10))
    plt.title(
        f"Models' accuracy performance for on ({num_facts}facts-{is_shuffle}shuffle):"
    )
    plt.xlabel("Number of modified facts")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(range(1, num_facts + 1))

    for model in plot_models:
        plt.plot(range(1, num_facts + 1), model_accuracy[model], "-o", label=model)

    plt.legend()
    plt.show()
    plt.savefig(f"result/figures/acc-{num_facts}facts-{is_shuffle}shuffle.png")


def write_overlap_res(models):
    for num_facts in [3, 4]:
        for is_shuffle in [0, 1]:
            all_qids = []
            for num_modified_facts in range(1, num_facts + 1):
                in_fn = f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{is_shuffle}shuffle-test_set.json"
                data = load_json(in_fn)

                qids = []
                for i, instance in enumerate(tqdm(data)):
                    qids.append(instance["qid"])

                all_qids.append(qids)
            overlap_qids = [i for i in all_qids[0] if i in all_qids[1]]
            for ix in range(2, num_facts):
                overlap_qids = [i for i in overlap_qids if i in all_qids[ix]]

            for num_modified_facts in range(1, num_facts + 1):
                for model in models:
                    new_res = []
                    in_fn = f"test/{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{is_shuffle}shuffle({model}).json"
                    out_fn = f"test/overlap_{num_facts}facts/strategyqa_{num_facts}facts_{num_modified_facts}mofified_{is_shuffle}shuffle({model}).json"
                    data = load_json(in_fn)
                    for i, instance in enumerate(tqdm(data)):
                        qid = instance["qid"]
                        if qid in overlap_qids:
                            new_res.append(instance)
                    write_json(out_fn, new_res)


if __name__ == "__main__":
    models = [
        "gpt4",
        "chatgpt",
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
    res = get_accuracy_table(models)
    res.to_csv("result/fc-intensity-accuracy.csv")

    # for num_facts in [3,4]:
    #     for is_shuffle in [0,1]:
    # res = get_accuracy_table(models)
    # print(res)
    # res.to_csv(f'result/tables/accuracy_modify.csv', index=False)

    # eva_llm_models = ["claude-v3-sonnet", "claude-v3-haiku","llama3-8b-instruct","llama3-70b-instruct","mixtral-8x7b"]
    # eva_as_models = ["as-base", "as-large"]
    # eva_nli_models = ["microsoft/deberta-xlarge-mnli", "microsoft/deberta-v2-xxlarge-mnli"]
    # eva_mini_models = ['roberta-large', 'deberta-v3-large', 'flan-t5-large']

    # for num_facts in [3,4]:
    #     for is_shuffle in [0,1]:
    #         res = {}
    #         for num_modified_facts in range(1, num_facts+1):
    #             a = {"evi vs. ori_evi":{}, "ori_evi vs. evi":{}}
    #             a = get_as_accuracy(num_facts, num_modified_facts, is_shuffle, eva_as_models, a)
    #             a = get_mini_accuracy(num_facts, num_modified_facts, is_shuffle, eva_mini_models, a)
    #             a = get_llm_accuracy(num_facts, num_modified_facts, is_shuffle, eva_llm_models, a)
    #             a = get_nli_accuracy(num_facts, num_modified_facts, is_shuffle, eva_nli_models, a)

    #             res[f"{num_modified_facts}modified"] = a
    #         write_json(f"result/{num_facts}facts_{is_shuffle}shuffle_singlepairs.json", res)

    # a = load_json('result/4facts_0shuffle_singlepairs.json')
    # for num_modified_facts in range(1,5):
    #     modify = f"{num_modified_facts}modified"
    #     a = get_llm_accuracy(4, num_modified_facts, 0, modify, ["gpt4"], a)
    # write_json(f"result/4facts_0shuffle_singlepairs.json", a)
