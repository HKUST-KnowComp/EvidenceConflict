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
    res_direct = {}

    # E1-E2 (no conflict)
    n_preds = [i["within_texts"][0] for i in data if i["within_texts"]]
    pred = aggregate_pred_llm(n_preds, eva_model)
    true = [0 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]

    # E1/E2-E3 (has conflict)
    y_preds = [i["within_texts"][1] for i in data if i["within_texts"]] + [
        i["within_texts"][2] for i in data if i["within_texts"]
    ]
    pred = aggregate_pred_llm(y_preds, eva_model)
    true.extend([1 for _ in range(len(pred["conditioned"][eva_model]))])
    for k in pred:
        for t in pred[k]:
            p.extend(pred[k][t])

    res_direct[t] = classification_report(true, p, output_dict=True)

    return res_direct


def get_llm_results_pollution(data, eva_model):
    res_pollution = {}

    # E1'-E1
    ApA_pred = [i["revision_texts"][0] for i in data if i["revision_texts"]]
    pred = aggregate_pred_llm(ApA_pred, eva_model)

    true = [1 for _ in range(len(pred["conditioned"][eva_model]))]
    for k in pred:
        for t in pred[k]:
            p = pred[k][t]

    # E1'-E2
    ApB_pred = [i["revision_texts"][1] for i in data if i["revision_texts"]]
    pred = aggregate_pred_llm(ApB_pred, eva_model)

    true.extend([1 for _ in range(len(pred["conditioned"][eva_model]))])
    for k in pred:
        for t in pred[k]:
            p.extend(pred[k][t])

    # E1'-E3
    ApC_pred = [i["revision_texts"][2] for i in data if i["revision_texts"]]
    pred = aggregate_pred_llm(ApC_pred, eva_model)

    true.extend([0 for _ in range(len(pred["conditioned"][eva_model]))])
    for k in pred:
        for t in pred[k]:
            p.extend(pred[k][t])

    res_pollution[t] = classification_report(true, p, output_dict=True)

    return res_pollution


def get_llm_accuracy(
    dataset, length, eva_llm_models, accuracy_rates={"direct": {}, "pollution": {}}
):
    for eva_llm_model in eva_llm_models:
        print(f"llm model is {eva_llm_model}")

        data = load_json(f"data/{dataset}-{length}({eva_llm_model}).json")
        if len(data) > 300:
            data = data[:300]

        res_direct = get_llm_results_direct(data, eva_llm_model)
        accuracy_rates["direct"].update(res_direct)

        res_pollution = get_llm_results_pollution(data, eva_llm_model)
        accuracy_rates["pollution"].update(res_pollution)

    return accuracy_rates


"""nli"""


def aggregate_pred_nli(all_preds):
    agg = {
        "c>e": [],
        "max": [],
    }

    for p in all_preds:
        for k in agg:
            agg[k].append(p[k])
    return agg


def get_nli_results_direct(data, model):
    res_direct = {}

    # E1-E2 (no conflict)
    n_preds = [i["within_texts"][0] for i in data if i["within_texts"]]
    n_preds = [i["conditioned"] for i in n_preds]

    pred = aggregate_pred_nli(n_preds)
    true = [0 for _ in range(len(pred["c>e"]))]

    p1 = pred["c>e"]
    p2 = pred["max"]

    # E1/E2-E3 (has conflict)
    y_preds = [i["within_texts"][1] for i in data if i["within_texts"]] + [
        i["within_texts"][2] for i in data if i["within_texts"]
    ]
    y_preds = [i["conditioned"] for i in y_preds]
    pred = aggregate_pred_nli(y_preds)
    true.extend([1 for _ in range(len(pred["c>e"]))])

    p1.extend(pred["c>e"])
    p2.extend(pred["max"])

    res_direct[f"{model}: c>e"] = classification_report(true, p1, output_dict=True)
    res_direct[f"{model}: max"] = classification_report(true, p2, output_dict=True)

    return res_direct


def get_nli_results_pollution(data, model):
    res_pollution = {}

    # E1'-E1
    ApA_pred = [i["revision_texts"][0] for i in data if i["revision_texts"]]
    ApA_pred = [i["conditioned"] for i in ApA_pred]
    pred = aggregate_pred_nli(ApA_pred)

    true = [1 for _ in range(len(pred["c>e"]))]

    p1 = pred["c>e"]
    p2 = pred["max"]

    # E1'-E2
    ApB_pred = [i["revision_texts"][1] for i in data if i["revision_texts"]]
    ApB_pred = [i["conditioned"] for i in ApB_pred]
    pred = aggregate_pred_nli(ApB_pred)

    true.extend([1 for _ in range(len(pred["c>e"]))])

    p1.extend(pred["c>e"])
    p2.extend(pred["max"])

    # E1'-E3
    ApC_pred = [i["revision_texts"][2] for i in data if i["revision_texts"]]
    ApC_pred = [i["conditioned"] for i in ApC_pred]
    pred = aggregate_pred_nli(ApC_pred)

    true.extend([0 for _ in range(len(pred["c>e"]))])

    p1.extend(pred["c>e"])
    p2.extend(pred["max"])

    res_pollution[f"{model}: c>e"] = classification_report(true, p1, output_dict=True)
    res_pollution[f"{model}: max"] = classification_report(true, p2, output_dict=True)

    return res_pollution


def get_nli_accuracy(
    dataset, length, eva_nli_models, accuracy_rates={"direct": {}, "pollution": {}}
):

    for eva_nli_model in eva_nli_models:
        print(f"nli model is {eva_nli_model}")

        data = load_json(
            f"data/{dataset}-{length}({eva_nli_model.replace('/','-')}).json"
        )
        if len(data) > 300:
            data = data[:300]

        model_name = eva_nli_model.replace("microsoft/", "")

        res_direct = get_nli_results_direct(data, model_name)
        accuracy_rates["direct"].update(res_direct)

        res_pollution = get_nli_results_pollution(data, model_name)
        accuracy_rates["pollution"].update(res_pollution)

    return accuracy_rates


"""'align score"""


def aggregate_pred_as(all_preds, eva_as_model):
    agg = {eva_as_model: []}

    for p in all_preds:
        for k in agg:
            agg[k].append(p[k])
    return agg


def get_as_results_direct(data, eva_model):
    res_direct = {}

    # E1-E2 (no conflict)
    n_preds = [i["within_texts"][0] for i in data if i["within_texts"]]
    n_preds = [i["conditioned"] for i in n_preds]

    pred = aggregate_pred_as(n_preds, eva_model)
    true = [0 for _ in range(len(pred[eva_model]))]
    for k in pred:
        p = pred[k]

    # E1/E2-E3 (has conflict)
    y_preds = [i["within_texts"][1] for i in data if i["within_texts"]] + [
        i["within_texts"][2] for i in data if i["within_texts"]
    ]
    y_preds = [i["conditioned"] for i in y_preds]

    pred = aggregate_pred_as(y_preds, eva_model)
    true.extend([1 for _ in range(len(pred[eva_model]))])
    for k in pred:
        p.extend(pred[k])

    res_direct[k] = classification_report(true, p, output_dict=True)

    return res_direct


def get_as_results_pollution(data, eva_model):
    res_pollution = {}

    # E1'-E1
    ApA_pred = [i["revision_texts"][0] for i in data if i["revision_texts"]]
    ApA_pred = [i["conditioned"] for i in ApA_pred]
    pred = aggregate_pred_as(ApA_pred, eva_model)

    true = [1 for _ in range(len(pred[eva_model]))]
    for k in pred:
        p = pred[k]

    # E1'-E2
    ApB_pred = [i["revision_texts"][1] for i in data if i["revision_texts"]]
    ApB_pred = [i["conditioned"] for i in ApB_pred]
    pred = aggregate_pred_as(ApB_pred, eva_model)

    true.extend([1 for _ in range(len(pred[eva_model]))])
    for k in pred:
        p.extend(pred[k])

    # E1'-E3
    res_3 = {}
    ApC_pred = [i["revision_texts"][2] for i in data if i["revision_texts"]]
    ApC_pred = [i["conditioned"] for i in ApC_pred]
    pred = aggregate_pred_as(ApC_pred, eva_model)

    true.extend([0 for _ in range(len(pred[eva_model]))])
    for k in pred:
        p.extend(pred[k])

    res_pollution[k] = classification_report(true, p, output_dict=True)

    return res_pollution


def get_as_accuracy(
    dataset, length, eva_as_models, accuracy_rates={"direct": {}, "pollution": {}}
):
    for eva_as_model in eva_as_models:
        print(f"as model is {eva_as_model}")

        data = load_json(f"data/{dataset}-{length}({eva_as_model}).json")
        if len(data) > 300:
            data = data[:300]

        res_direct = get_as_results_direct(data, eva_as_model)
        accuracy_rates["direct"].update(res_direct)

        res_pollution = get_as_results_pollution(data, eva_as_model)
        accuracy_rates["pollution"].update(res_pollution)

    return accuracy_rates


if __name__ == "__main__":
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
    eva_as_models = ["as-base", "as-large"]

    for dataset in ["nq_open", "cwq"]:
        for length in ["short", "long"]:
            a = get_llm_accuracy(dataset, length, eva_llm_models)
            a = get_nli_accuracy(dataset, length, eva_nli_models, a)
            a = get_as_accuracy(dataset, length, eva_as_models, a)
            write_json(f"results/{dataset}-{length}_results_v2.json", a)
