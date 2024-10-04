import json


def load_json(fn):
    with open(fn) as f:
        return json.loads(f.read())


def load_jsonl(fn):
    data = []
    with open(fn) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def write_json(fn, data):
    with open(fn, "w") as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))


def get_llm_response(prompt, model="chatgpt", max_new_tokens=1024, stop_sequences=[]):
    """Prompting LLMs to get response.
    prompt <str>: the input text to be fed to the LLMs.
    model <str>: the model name.

    This method outputs the model generation (string).

    You should implement this part to proceed.
    """
    raise NotImplementedError(
        "Method 'get_llm_response' in 'utils.py' is not yet implemented!"
    )


if __name__ == "__main__":
    res = get_llm_response("i am eating an apple", model="mistral-7b")
    print(res)
