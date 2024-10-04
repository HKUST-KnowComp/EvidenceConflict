from sklearn.metrics import classification_report
from utils import load_json, write_json
import pandas as pd
import numpy as np

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


def get_llm_results(data, eva_model):
    print("data length:", len(data))

    # direct
    all_preds, all_trues = [], []

    # E1-E2 (no conflict)
    n_preds = [i["within_texts"][0] for i in data if i["within_texts"]]
    pred = aggregate_pred_llm(n_preds, eva_model)
    true = [0 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            all_preds.extend(p)
            all_trues.extend(true)
    print("E1-E2", len(all_trues), len(all_preds))
    # E1/E2-E3 (has conflict)
    y_preds = [i["within_texts"][1] for i in data if i["within_texts"]] + [
        i["within_texts"][2] for i in data if i["within_texts"]
    ]
    pred = aggregate_pred_llm(y_preds, eva_model)
    true = [1 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            all_preds.extend(p)
            all_trues.extend(true)

    print("direct", len(all_trues), len(all_preds))
    res_direct = classification_report(all_trues, all_preds, output_dict=True)

    # pollution
    all_preds, all_trues = [], []

    # E1'-E1
    ApA_pred = [i["revision_texts"][0] for i in data if i["revision_texts"]]
    pred = aggregate_pred_llm(ApA_pred, eva_model)

    true = [1 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            all_preds.extend(p)
            all_trues.extend(true)
    print("E1'-E1", len(all_trues), len(all_preds))
    # E1'-E2
    ApB_pred = [i["revision_texts"][1] for i in data if i["revision_texts"]]
    pred = aggregate_pred_llm(ApB_pred, eva_model)

    true = [1 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            all_preds.extend(p)
            all_trues.extend(true)
    print("E1'-E1,E1'-E2", len(all_trues), len(all_preds))
    # E1'-E3
    ApC_pred = [i["revision_texts"][2] for i in data if i["revision_texts"]]
    pred = aggregate_pred_llm(ApC_pred, eva_model)

    true = [0 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]
            all_preds.extend(p)
            all_trues.extend(true)
    print("pollution", len(all_trues), len(all_preds))
    res_pollution = classification_report(all_trues, all_preds, output_dict=True)

    return res_direct, res_pollution


def get_llm_accuracy(
    dataset,
    length,
    eva_llm_models,
    reverse="",
    accuracy_rates={"direct": {}, "pollution": {}},
):
    for eva_llm_model in eva_llm_models:
        print(f"llm model is {eva_llm_model}")

        data = load_json(f"data/{dataset}-{length}({eva_llm_model}){reverse}.json")

        if len(data) > 300:
            data = data[:300]

        res_d, res_p = get_llm_results(data, eva_llm_model)

        accuracy_rates["direct"][eva_llm_model] = res_d
        accuracy_rates["pollution"][eva_llm_model] = res_p

    return accuracy_rates


"""nli"""


def aggregate_pred_nli(all_preds, eva_model, type_name):
    agg = {
        "conditioned": {
            type_name: [],
        }
    }

    for p in all_preds:
        for k in agg:
            if k == "conditioned":
                for t in agg[k]:
                    if t == type_name:
                        agg[k][t].append(p[k][t])
    return agg["conditioned"][type_name]


def get_nli_results(data, eva_model, type_name):

    # direct
    all_preds, all_trues = [], []

    # E1-E2 (no conflict)
    n_preds = [i["within_texts"][0] for i in data if i["within_texts"]]
    pred = aggregate_pred_nli(n_preds, eva_model, type_name)
    true = [0 for _ in range(len(pred))]

    all_preds.extend(pred)
    all_trues.extend(true)
    print("E1-E2", len(all_trues), len(all_preds))
    # E1/E2-E3 (has conflict)
    y_preds = [i["within_texts"][1] for i in data if i["within_texts"]] + [
        i["within_texts"][2] for i in data if i["within_texts"]
    ]
    pred = aggregate_pred_nli(y_preds, eva_model, type_name)
    true = [1 for _ in range(len(pred))]

    all_preds.extend(pred)
    all_trues.extend(true)
    print("direct", len(all_trues), len(all_preds))
    res_direct = classification_report(all_trues, all_preds, output_dict=True)

    # pollution
    all_preds, all_trues = [], []

    # E1'-E1
    ApA_pred = [i["revision_texts"][0] for i in data if i["revision_texts"]]
    pred = aggregate_pred_nli(ApA_pred, eva_model, type_name)
    true = [1 for _ in range(len(pred))]

    all_preds.extend(pred)
    all_trues.extend(true)
    print("E1'-E1", len(all_trues), len(all_preds))
    # E1'-E2
    ApB_pred = [i["revision_texts"][1] for i in data if i["revision_texts"]]
    pred = aggregate_pred_nli(ApB_pred, eva_model, type_name)
    true = [1 for _ in range(len(pred))]

    all_preds.extend(pred)
    all_trues.extend(true)
    print("E1'-E2", len(all_trues), len(all_preds))
    # E1'-E3
    ApC_pred = [i["revision_texts"][2] for i in data if i["revision_texts"]]
    pred = aggregate_pred_nli(ApC_pred, eva_model, type_name)
    true = [0 for _ in range(len(pred))]

    all_preds.extend(pred)
    all_trues.extend(true)
    print("pollution", len(all_trues), len(all_preds))
    res_pollution = classification_report(all_trues, all_preds, output_dict=True)

    return res_direct, res_pollution


def get_nli_accuracy(
    dataset,
    length,
    eva_nli_models,
    reverse="",
    accuracy_rates={"direct": {}, "pllution": {}},
):

    for eva_nli_model in eva_nli_models:
        print(f"nli model is {eva_nli_model}")

        data = load_json(
            f"data/{dataset}-{length}({eva_nli_model.replace('/','-')}){reverse}.json"
        )

        if len(data) > 300:
            data = data[:300]

        for type_name in ["c>e", "max"]:
            res_d, res_p = get_nli_results(
                data, eva_nli_model.replace("/", "-"), type_name
            )
            accuracy_rates["direct"][
                f"{eva_nli_model.replace('microsoft/','')}: {type_name}"
            ] = res_d
            accuracy_rates["pollution"][
                f"{eva_nli_model.replace('microsoft/','')}: {type_name}"
            ] = res_p

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
    return agg["conditioned"][eva_as_model]


def get_as_results(data, eva_model):

    # direct
    all_trues, all_preds = [], []

    # E1-E2 (no conflict)
    n_preds = [i["within_texts"][0] for i in data if i["within_texts"]]
    pred = aggregate_pred_as(n_preds, eva_model)
    true = [0 for _ in range(len(pred))]
    all_preds.extend(pred)
    all_trues.extend(true)

    # E1/E2-E3 (has conflict)
    y_preds = [i["within_texts"][1] for i in data if i["within_texts"]] + [
        i["within_texts"][2] for i in data if i["within_texts"]
    ]
    pred = aggregate_pred_as(y_preds, eva_model)
    true = [1 for _ in range(len(pred))]
    all_preds.extend(pred)
    all_trues.extend(true)
    print(len(all_trues), len(all_preds))
    res_direct = classification_report(all_trues, all_preds, output_dict=True)

    # pollution
    all_trues, all_preds = [], []

    # E1'-E1
    ApA_pred = [i["revision_texts"][0] for i in data if i["revision_texts"]]
    pred = aggregate_pred_as(ApA_pred, eva_model)
    true = [1 for _ in range(len(pred))]
    all_preds.extend(pred)
    all_trues.extend(true)

    # E1'-E2
    ApB_pred = [i["revision_texts"][1] for i in data if i["revision_texts"]]
    pred = aggregate_pred_as(ApB_pred, eva_model)
    true = [1 for _ in range(len(pred))]
    all_preds.extend(pred)
    all_trues.extend(true)

    # E1'-E3
    ApC_pred = [i["revision_texts"][2] for i in data if i["revision_texts"]]
    pred = aggregate_pred_as(ApC_pred, eva_model)
    true = [0 for _ in range(len(pred))]
    all_preds.extend(pred)
    all_trues.extend(true)
    print(len(all_trues), len(all_preds))
    res_pollution = classification_report(all_trues, all_preds, output_dict=True)

    return res_direct, res_pollution


def get_as_accuracy(
    dataset,
    length,
    eva_as_models,
    reverse="",
    accuracy_rates={"direct": {}, "pollution": {}},
):
    for eva_as_model in eva_as_models:
        print(f"as model is {eva_as_model}")

        data = load_json(f"data/{dataset}-{length}({eva_as_model}){reverse}.json")

        if len(data) > 300:
            data = data[:300]

        res_d, res_p = get_as_results(data, eva_as_model)
        accuracy_rates["direct"][eva_as_model] = res_d
        accuracy_rates["pollution"][eva_as_model] = res_p

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
    return agg["conditioned"][eva_mini_model]


def get_mini_results(data, eva_model):

    # direct
    all_preds, all_trues = [], []
    # E1-E2 (no conflict)
    n_preds = [i["within_texts"][0] for i in data if i["within_texts"]]
    pred = aggregate_pred_mini(n_preds, eva_model)
    true = [0 for _ in range(len(pred))]
    all_preds.extend(pred)
    all_trues.extend(true)

    # E1/E2-E3 (has conflict)
    y_preds = [i["within_texts"][1] for i in data if i["within_texts"]] + [
        i["within_texts"][2] for i in data if i["within_texts"]
    ]
    pred = aggregate_pred_mini(y_preds, eva_model)
    true = [1 for _ in range(len(pred))]
    all_preds.extend(pred)
    all_trues.extend(true)
    print(len(all_trues), len(all_preds))
    res_direct = classification_report(all_trues, all_preds, output_dict=True)

    # pollution
    all_preds, all_trues = [], []

    # E1'-E1
    ApA_pred = [i["revision_texts"][0] for i in data if i["revision_texts"]]
    pred = aggregate_pred_mini(ApA_pred, eva_model)
    true = [1 for _ in range(len(pred))]
    all_preds.extend(pred)
    all_trues.extend(true)

    # E1'-E2
    ApB_pred = [i["revision_texts"][1] for i in data if i["revision_texts"]]
    pred = aggregate_pred_mini(ApB_pred, eva_model)
    true = [1 for _ in range(len(pred))]
    all_preds.extend(pred)
    all_trues.extend(true)

    # E1'-E3
    ApC_pred = [i["revision_texts"][2] for i in data if i["revision_texts"]]
    pred = aggregate_pred_mini(ApC_pred, eva_model)
    true = [0 for _ in range(len(pred))]
    all_preds.extend(pred)
    all_trues.extend(true)
    print(len(all_trues), len(all_preds))
    res_pollution = classification_report(all_trues, all_preds, output_dict=True)

    return res_direct, res_pollution


def get_mini_accuracy(
    dataset,
    length,
    eva_mini_models,
    reverse="",
    accuracy_rates={"direct": {}, "pollution": {}},
):
    for eva_mini_model in eva_mini_models:
        print(f"as model is {eva_mini_model}")

        data = load_json(f"data/{dataset}-{length}({eva_mini_model}){reverse}.json")

        if len(data) > 300:
            data = data[:300]

        res_d, res_p = get_mini_results(data, eva_mini_model)
        accuracy_rates["direct"][eva_mini_model] = res_d
        accuracy_rates["pollution"][eva_mini_model] = res_p

    return accuracy_rates


"""get table"""


def get_result_table():
    res = pd.DataFrame()

    models = [
        "mixtral-8x7b",
        "llama3-8b-instruct",
        "llama3-70b-instruct",
        "claude-v3-haiku",
        "claude-v3-sonnet",
        "chatgpt",
        "gpt4",
        "as-base",
        "as-large",
        "roberta-large",
        "deberta-v3-large",
        "flan-t5-large",
        "deberta-xlarge-mnli: c>e",
        "deberta-xlarge-mnli: max",
        "deberta-v2-xxlarge-mnli: c>e",
        "deberta-v2-xxlarge-mnli: max",
    ]

    p_mean, r_mean, f_mean = [], [], []
    for dataset in ["nq_open", "cwq"]:
        for length in ["short", "long"]:
            p, r, f = [], [], []
            for reverse in ["", "_reverse"]:
                ptmp, rtmp, ftmp = [], [], []

                in_fn = (
                    f"results/pooled/{dataset}_{length}{reverse}-accuracy-pooled.json"
                )
                data = load_json(in_fn)

                for model in models:
                    ptmp.append(
                        round(data["direct"][model]["macro avg"]["precision"] * 100, 2)
                    )
                    rtmp.append(
                        round(data["direct"][model]["macro avg"]["recall"] * 100, 2)
                    )
                    ftmp.append(
                        round(data["direct"][model]["macro avg"]["f1-score"] * 100, 2)
                    )

                p.append(ptmp)
                r.append(rtmp)
                f.append(ftmp)

            p = [round(i, 2) for i in np.mean(np.array(p), axis=0)]
            r = [round(i, 2) for i in np.mean(np.array(r), axis=0)]
            f = [round(i, 2) for i in np.mean(np.array(f), axis=0)]

            res.loc[:, f"{dataset}-{length}-precision"] = p
            res.loc[:, f"{dataset}-{length}-recall"] = r
            res.loc[:, f"{dataset}-{length}-f1_score"] = f

            p_mean.append(p)
            r_mean.append(r)
            f_mean.append(f)

    p_mean = [round(i, 2) for i in np.mean(np.array(p_mean), axis=0)]
    r_mean = [round(i, 2) for i in np.mean(np.array(r_mean), axis=0)]
    f_mean = [round(i, 2) for i in np.mean(np.array(f_mean), axis=0)]

    res.loc[:, "mean-precision"] = p_mean
    res.loc[:, "mean-recall"] = r_mean
    res.loc[:, "mean-f1_score"] = f_mean

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

    res["model"] = models
    res["model"] = res["model"].apply(lambda x: model_map[x])
    res["model"] = pd.Categorical(res["model"], categories=all_models, ordered=True)
    res = res.groupby(["model"]).first()

    return res


if __name__ == "__main__":

    # eva_nli_models = ["microsoft/deberta-xlarge-mnli", "microsoft/deberta-v2-xxlarge-mnli"]
    # eva_llm_models = ["claude-v3-sonnet", "claude-v3-haiku","llama3-8b-instruct","llama3-70b-instruct","mixtral-8x7b"]
    # eva_as_models = ["as-base","as-large"]
    # eva_mini_models = ['roberta-large', 'deberta-v3-large', 'flan-t5-large']

    for dataset in ["cwq", "nq_open"]:
        for length in ["short", "long"]:
            for reverse in ["", "_reverse"]:
                print("-" * 30)
                print(dataset, length, reverse)
                a = load_json(
                    f"results/pooled/{dataset}_{length}{reverse}-accuracy-pooled.json"
                )
                a = get_llm_accuracy(dataset, length, ["gpt4"], reverse, a)

                # # #             a = {'direct':{}, 'pollution':{}}
                # #             a = get_nli_accuracy(dataset, length, eva_nli_models, reverse, a)
                # #             a = get_llm_accuracy(dataset, length, eva_llm_models, reverse, a)
                # #             a = get_as_accuracy(dataset, length, eva_as_models, reverse, a)
                # #             a = get_mini_accuracy(dataset, length, eva_mini_models, reverse, a)
                write_json(
                    f"results/pooled/{dataset}_{length}{reverse}-accuracy-pooled.json",
                    a,
                )

    res = get_result_table()
    res.to_csv("results/ac_pooled_accuracy.csv")
