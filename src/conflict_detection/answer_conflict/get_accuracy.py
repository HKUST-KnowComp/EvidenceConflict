import pandas as pd
from sklearn.metrics import classification_report
from utils import load_json, write_json

"""llm"""


def aggregate_pred_llm(all_preds, eva_llm_model):
    res = {}
    res[eva_llm_model] = []
    agg = {"conditioned": res}

    for p in all_preds:
        for k in agg:
            for t in agg[k]:
                agg[k][t].append(p[k][t])
    return agg


def get_llm_results_direct(data, eva_model):

    # E1-E2 (no conflict)
    res_n = {}
    n_preds = [i["within_texts"][0] for i in data if i["within_texts"]]
    pred = aggregate_pred_llm(n_preds, eva_model)
    true = [0 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_n[t] = classification_report(true, p, output_dict=True)["accuracy"]

    # E1/E2-E3 (has conflict)
    res_y = {}
    y_preds = [i["within_texts"][1] for i in data if i["within_texts"]] + [
        i["within_texts"][2] for i in data if i["within_texts"]
    ]
    pred = aggregate_pred_llm(y_preds, eva_model)
    true = [1 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_y[t] = classification_report(true, p, output_dict=True)["accuracy"]
    return res_n, res_y


def get_llm_results_pollution(data, eva_model):

    # E1'-E1
    res_1 = {}
    ApA_pred = [i["revision_texts"][0] for i in data if i["revision_texts"]]
    pred = aggregate_pred_llm(ApA_pred, eva_model)

    true = [1 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_1[t] = classification_report(true, p, output_dict=True)["accuracy"]

    # E1'-E2
    res_2 = {}
    ApB_pred = [i["revision_texts"][1] for i in data if i["revision_texts"]]
    pred = aggregate_pred_llm(ApB_pred, eva_model)

    true = [1 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_2[t] = classification_report(true, p, output_dict=True)["accuracy"]

    # E1'-E3
    res_3 = {}
    ApC_pred = [i["revision_texts"][2] for i in data if i["revision_texts"]]
    pred = aggregate_pred_llm(ApC_pred, eva_model)

    true = [0 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_3[t] = classification_report(true, p, output_dict=True)["accuracy"]

    return res_1, res_2, res_3


def get_llm_accuracy(
    dataset,
    length,
    eva_llm_models,
    reverse="",
    accuracy_rates={
        "direct": {"E1-E2": {}, "E1,E2-E3": {}},
        "pollution": {"E1'-E1": {}, "E1'-E2": {}, "E1'-E3": {}},
    },
):
    for eva_llm_model in eva_llm_models:
        print(f"llm model is {eva_llm_model}")

        data = load_json(f"data/{dataset}-{length}({eva_llm_model}){reverse}.json")

        if len(data) > 300:
            data = data[:300]

        res_n, res_y = get_llm_results_direct(data, eva_llm_model)
        accuracy_rates["direct"]["E1-E2"].update(res_n)
        accuracy_rates["direct"]["E1,E2-E3"].update(res_y)

        res_1, res_2, res_3 = get_llm_results_pollution(data, eva_llm_model)
        accuracy_rates["pollution"]["E1'-E1"].update(res_1)
        accuracy_rates["pollution"]["E1'-E2"].update(res_2)
        accuracy_rates["pollution"]["E1'-E3"].update(res_3)

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


def get_nli_results_direct(data, type_name):

    # E1-E2 (no conflict)
    res_n = {}
    n_preds = [i["within_texts"][0] for i in data if i["within_texts"]]
    pred = aggregate_pred_nli(n_preds, type_name)
    true = [0 for _ in range(len(pred["conditioned"][type_name]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_n = classification_report(true, p, output_dict=True)["accuracy"]

    # E1/E2-E3 (has conflict)
    res_y = {}
    y_preds = [i["within_texts"][1] for i in data if i["within_texts"]] + [
        i["within_texts"][2] for i in data if i["within_texts"]
    ]
    pred = aggregate_pred_nli(y_preds, type_name)
    true = [1 for _ in range(len(pred["conditioned"][type_name]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_y = classification_report(true, p, output_dict=True)["accuracy"]

    return res_n, res_y


def get_nli_results_pollution(data, type_name):

    # E1'-E1
    res_1 = {}
    ApA_pred = [i["revision_texts"][0] for i in data if i["revision_texts"]]
    pred = aggregate_pred_nli(ApA_pred, type_name)

    true = [1 for _ in range(len(pred["conditioned"][type_name]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_1 = classification_report(true, p, output_dict=True)["accuracy"]

    # E1'-E2
    res_2 = {}
    ApB_pred = [i["revision_texts"][1] for i in data if i["revision_texts"]]
    pred = aggregate_pred_nli(ApB_pred, type_name)

    true = [1 for _ in range(len(pred["conditioned"][type_name]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_2 = classification_report(true, p, output_dict=True)["accuracy"]

    # E1'-E3
    res_3 = {}
    ApC_pred = [i["revision_texts"][2] for i in data if i["revision_texts"]]
    pred = aggregate_pred_nli(ApC_pred, type_name)

    true = [0 for _ in range(len(pred["conditioned"][type_name]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_3 = classification_report(true, p, output_dict=True)["accuracy"]

    return res_1, res_2, res_3


def get_nli_accuracy(
    dataset,
    length,
    eva_nli_models,
    reverse="",
    accuracy_rates={
        "direct": {"E1-E2": {}, "E1,E2-E3": {}},
        "pollution": {"E1'-E1": {}, "E1'-E2": {}, "E1'-E3": {}},
    },
):

    for eva_nli_model in eva_nli_models:
        print(f"nli model is {eva_nli_model}")

        data = load_json(
            f"data/{dataset}-{length}({eva_nli_model.replace('/','-')}){reverse}.json"
        )

        if len(data) > 300:
            data = data[:300]

        for type_name in ["c>e", "max"]:
            res_n, res_y = get_nli_results_direct(data, type_name)
            accuracy_rates["direct"]["E1-E2"].update(
                {f'{eva_nli_model.replace("microsoft/","")}: {type_name}': res_n}
            )
            accuracy_rates["direct"]["E1,E2-E3"].update(
                {f'{eva_nli_model.replace("microsoft/","")}: {type_name}': res_y}
            )

            res_1, res_2, res_3 = get_nli_results_pollution(data, type_name)
            accuracy_rates["pollution"]["E1'-E1"].update(
                {f'{eva_nli_model.replace("microsoft/","")}: {type_name}': res_1}
            )
            accuracy_rates["pollution"]["E1'-E2"].update(
                {f'{eva_nli_model.replace("microsoft/","")}: {type_name}': res_2}
            )
            accuracy_rates["pollution"]["E1'-E3"].update(
                {f'{eva_nli_model.replace("microsoft/","")}: {type_name}': res_3}
            )

    return accuracy_rates


"""align score"""


def aggregate_pred_as(all_preds, eva_as_model):
    res = {}
    res[eva_as_model] = []
    agg = {"conditioned": res}

    for p in all_preds:
        for k in agg:
            if k == "conditioned":
                for t in agg[k]:
                    agg[k][t].append(p[k][t])
    return agg


def get_as_results_direct(data, eva_model):

    # E1-E2 (no conflict)
    res_n = {}
    n_preds = [i["within_texts"][0] for i in data if i["within_texts"]]
    pred = aggregate_pred_as(n_preds, eva_model)
    true = [0 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_n[eva_model] = classification_report(true, p, output_dict=True)[
                "accuracy"
            ]

    # E1/E2-E3 (has conflict)
    res_y = {}
    y_preds = [i["within_texts"][1] for i in data if i["within_texts"]] + [
        i["within_texts"][2] for i in data if i["within_texts"]
    ]
    pred = aggregate_pred_as(y_preds, eva_model)
    true = [1 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_y[eva_model] = classification_report(true, p, output_dict=True)[
                "accuracy"
            ]

    return res_n, res_y


def get_as_results_pollution(data, eva_model):

    # E1'-E1
    res_1 = {}
    ApA_pred = [i["revision_texts"][0] for i in data if i["revision_texts"]]
    pred = aggregate_pred_as(ApA_pred, eva_model)

    true = [1 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_1[eva_model] = classification_report(true, p, output_dict=True)[
                "accuracy"
            ]

    # E1'-E2
    res_2 = {}
    ApB_pred = [i["revision_texts"][1] for i in data if i["revision_texts"]]
    pred = aggregate_pred_as(ApB_pred, eva_model)

    true = [1 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_2[eva_model] = classification_report(true, p, output_dict=True)[
                "accuracy"
            ]

    # E1'-E3
    res_3 = {}
    ApC_pred = [i["revision_texts"][2] for i in data if i["revision_texts"]]
    pred = aggregate_pred_as(ApC_pred, eva_model)

    true = [0 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_3[eva_model] = classification_report(true, p, output_dict=True)[
                "accuracy"
            ]

    return res_1, res_2, res_3


def get_as_accuracy(
    dataset,
    length,
    eva_as_models,
    reverse="",
    accuracy_rates={
        "direct": {"E1-E2": {}, "E1,E2-E3": {}},
        "pollution": {"E1'-E1": {}, "E1'-E2": {}, "E1'-E3": {}},
    },
):
    for eva_as_model in eva_as_models:
        print(f"as model is {eva_as_model}")

        data = load_json(f"data/{dataset}-{length}({eva_as_model}){reverse}.json")

        if len(data) > 300:
            data = data[:300]

        res_n, res_y = get_as_results_direct(data, eva_as_model)
        accuracy_rates["direct"]["E1-E2"].update(res_n)
        accuracy_rates["direct"]["E1,E2-E3"].update(res_y)

        res_1, res_2, res_3 = get_as_results_pollution(data, eva_as_model)
        accuracy_rates["pollution"]["E1'-E1"].update(res_1)
        accuracy_rates["pollution"]["E1'-E2"].update(res_2)
        accuracy_rates["pollution"]["E1'-E3"].update(res_3)

    return accuracy_rates


"""minicheck"""


def aggregate_pred_mini(all_preds, eva_mini_model):
    res = {}
    res[eva_mini_model] = []
    agg = {"conditioned": res}

    for p in all_preds:
        for k in agg:
            if k == "conditioned":
                for t in agg[k]:
                    agg[k][t].append(p[k][t])
    return agg


def get_mini_results_direct(data, eva_model):

    # E1-E2 (no conflict)
    res_n = {}
    n_preds = [i["within_texts"][0] for i in data if i["within_texts"]]
    pred = aggregate_pred_mini(n_preds, eva_model)
    true = [0 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_n[eva_model] = classification_report(true, p, output_dict=True)[
                "accuracy"
            ]

    # E1/E2-E3 (has conflict)
    res_y = {}
    y_preds = [i["within_texts"][1] for i in data if i["within_texts"]] + [
        i["within_texts"][2] for i in data if i["within_texts"]
    ]
    pred = aggregate_pred_mini(y_preds, eva_model)
    true = [1 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_y[eva_model] = classification_report(true, p, output_dict=True)[
                "accuracy"
            ]

    return res_n, res_y


def get_mini_results_pollution(data, eva_model):

    # E1'-E1
    res_1 = {}
    ApA_pred = [i["revision_texts"][0] for i in data if i["revision_texts"]]
    pred = aggregate_pred_mini(ApA_pred, eva_model)

    true = [1 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_1[eva_model] = classification_report(true, p, output_dict=True)[
                "accuracy"
            ]

    # E1'-E2
    res_2 = {}
    ApB_pred = [i["revision_texts"][1] for i in data if i["revision_texts"]]
    pred = aggregate_pred_mini(ApB_pred, eva_model)

    true = [1 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_2[eva_model] = classification_report(true, p, output_dict=True)[
                "accuracy"
            ]

    # E1'-E3
    res_3 = {}
    ApC_pred = [i["revision_texts"][2] for i in data if i["revision_texts"]]
    pred = aggregate_pred_mini(ApC_pred, eva_model)

    true = [0 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            res_3[eva_model] = classification_report(true, p, output_dict=True)[
                "accuracy"
            ]

    return res_1, res_2, res_3


def get_mini_accuracy(
    dataset,
    length,
    eva_mini_models,
    reverse="",
    accuracy_rates={
        "direct": {"E1-E2": {}, "E1,E2-E3": {}},
        "pollution": {"E1'-E1": {}, "E1'-E2": {}, "E1'-E3": {}},
    },
):
    for eva_mini_model in eva_mini_models:
        print(f"as model is {eva_mini_model}")

        data = load_json(f"data/{dataset}-{length}({eva_mini_model}){reverse}.json")

        if len(data) > 300:
            data = data[:300]

        res_n, res_y = get_mini_results_direct(data, eva_mini_model)
        accuracy_rates["direct"]["E1-E2"].update(res_n)
        accuracy_rates["direct"]["E1,E2-E3"].update(res_y)

        res_1, res_2, res_3 = get_mini_results_pollution(data, eva_mini_model)
        accuracy_rates["pollution"]["E1'-E1"].update(res_1)
        accuracy_rates["pollution"]["E1'-E2"].update(res_2)
        accuracy_rates["pollution"]["E1'-E3"].update(res_3)

    return accuracy_rates


"""aggregate to table"""
import pandas as pd
import numpy as np


def get_acc_table():
    accuracy_df = pd.DataFrame()

    for dataset in ["nq_open", "cwq"]:
        for length in ["short", "long"]:
            for reverse in ["", "_reverse"]:
                data = load_json(
                    f"results/breakdown/{dataset}_{length}{reverse}-accuracy.json"
                )
                acc_df_tmp = pd.DataFrame()

                # direct
                for pair in ["E1-E2", "E1,E2-E3"]:
                    tmp = pd.json_normalize(data["direct"][pair])
                    tmp["method"] = "direct"
                    tmp["pair"] = "E1-E2(N)" if pair == "E1-E2" else "E1,E2-E3(Y)"
                    acc_df_tmp = pd.concat([acc_df_tmp, tmp])

                # pollution
                for pair in ["E1'-E1", "E1'-E2", "E1'-E3"]:
                    tmp = pd.json_normalize(data["pollution"][pair])
                    tmp["method"] = "pollution"
                    tmp["pair"] = (
                        "E1'-E1(Y)"
                        if pair == "E1'-E1"
                        else ("E1'-E2(Y)" if pair == "E1'-E2" else "E1'-E3(N)")
                    )
                    acc_df_tmp = pd.concat([acc_df_tmp, tmp])

                acc_df_tmp["dataset"] = f"{dataset}"
                acc_df_tmp["length"] = "sentence" if length == "short" else "paragraph"

                accuracy_df = pd.concat([accuracy_df, acc_df_tmp])

    accuracy_df.to_csv("results/ac_accuracy.csv", index=False)


def plot_for_dataset_length():
    data = pd.read_csv("results/accuracy_table.csv")
    for col in data.columns.tolist():
        if "unconditioned" in col:
            data = data.drop(columns=col)
        if "conditioned" in col:
            data = data.rename(
                columns={
                    col: col.replace("conditioned", "").replace(".", "").strip("-")
                }
            )
        if "average" in col:
            data = data.rename(columns={col: col.replace("average", "").strip("-")})

    import matplotlib.pyplot as plt

    for length in ["sentence", "paragraph"]:
        for dataset in ["nq_open", "cwq"]:
            print(length, dataset)

            tmp = data[(data["length"] == length) & (data["dataset"] == dataset)]

            plt.figure(figsize=(10, 10))
            for model in [
                "deberta-large-mnli",
                "deberta-v2-xxlarge-mnli",
                "as-base",
                "as-large",
                "claude-v3-sonnet",
                "claude-v3-haiku",
                "llama3-8b-instruct",
                "llama3-70b-instruct",
                "mixtral-8x7b",
            ]:
                plt.plot(tmp["pair"], tmp[model], label=model)

            plt.title(f"Model Performance on {length}-level {dataset} dataset")
            plt.legend()
            plt.show()


def order():
    models = [
        "roberta-large",
        "deberta-v3-large",
        "flan-t5-large",
        "as-base",
        "as-large",
        "claude-v3-sonnet",
        "claude-v3-haiku",
        "llama3-8b-instruct",
        "llama3-70b-instruct",
        "mixtral-8x7b",
        "chatgpt",
        "gpt4",
        "deberta-xlarge-mnli: c>e",
        "deberta-v2-xxlarge-mnli: c>e",
        "deberta-xlarge-mnli: max",
        "deberta-v2-xxlarge-mnli: max",
    ]

    orderdf = pd.DataFrame()
    orderdf["model"] = models
    for dataset in ["nq_open", "cwq"]:
        for length in ["short", "long"]:
            for reverse in ["", "_reverse"]:
                res = load_json(
                    f"results/pooled/{dataset}_{length}{reverse}-accuracy-pooled.json"
                )
                p, r, f = [], [], []
                for model in models:
                    p.append(res["direct"][model]["macro avg"]["precision"] * 100)
                    r.append(res["direct"][model]["macro avg"]["recall"] * 100)
                    f.append(res["direct"][model]["macro avg"]["f1-score"] * 100)

                orderdf.loc[:, f"{dataset}_{length}{reverse}-P"] = p
                orderdf.loc[:, f"{dataset}_{length}{reverse}-R"] = r
                orderdf.loc[:, f"{dataset}_{length}{reverse}-F"] = f

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

    orderdf["model"] = orderdf["model"].apply(lambda x: model_map[x])
    orderdf["model"] = pd.Categorical(
        orderdf["model"], categories=all_models, ordered=True
    )

    orderdf = orderdf.groupby(["model"]).first()

    return orderdf


if __name__ == "__main__":

    # eva_nli_models = ["microsoft/deberta-xlarge-mnli", "microsoft/deberta-v2-xxlarge-mnli"]
    # eva_llm_models = ["claude-v3-sonnet", "claude-v3-haiku","llama3-8b-instruct","llama3-70b-instruct","mixtral-8x7b"]
    # eva_as_models = ["as-base","as-large"]
    # eva_mini_models = ['roberta-large', 'deberta-v3-large', 'flan-t5-large']

    # for dataset in ["cwq","nq_open"]:
    #     for length in ["short", "long"]:
    #         for reverse in ["", "_reverse"]:
    #             a = load_json(f"results/breakdown/{dataset}_{length}{reverse}-accuracy.json")
    #             a = get_llm_accuracy(dataset, length, ["gpt4"], reverse, a)

    # #             # a = accuracy_rates={"direct":{"E1-E2":{},"E1,E2-E3":{}},'pollution':{"E1'-E1":{},"E1'-E2":{},"E1'-E3":{}}}
    # #             # a = get_nli_accuracy(dataset, length, eva_nli_models, reverse, a)
    # #             # a = get_llm_accuracy(dataset, length, eva_llm_models, reverse, a)
    # #             # a = get_as_accuracy(dataset, length, eva_as_models, reverse, a)
    # #             # a = get_mini_accuracy(dataset, length, eva_mini_models, reverse, a)
    #             write_json(f"results/breakdown/{dataset}_{length}{reverse}-accuracy.json", a)
    res = order()
    print(res)
    res.to_csv("results/order.csv")
