from typing import Dict, List, Any, TypedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
import re
import torch
import streamlit as st
import streamlit.components.v1 as components

device = "cuda"

FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"


sample_code = """prompts = []
suggestion_index_to_token_index = []

for i, token in enumerate(tokens):
    if token.isspace() == "":
        # Don't generate suggestions for whitespace
        continue
    else:
        prefix = ''.join(tokens[:i])
        suffix = ''.join(tokens[i + 1 :])
        prompt = FIM_PREFIX + prefix + FIM_SUFFIX + suffix + FIM_MIDDLE

        prompts.append(prompt)
        suggestion_index_to_token_index.append(i)"""


@st.cache(allow_output_mutation=True)
def get_tokenizer():
    print("Loading tokenizer..")
    tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder", padding_side="left")
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                EOD,
                FIM_PREFIX,
                FIM_MIDDLE,
                FIM_SUFFIX,
                FIM_PAD,
            ],
            "eos_token": EOD,
            "pad_token": FIM_PAD,
        }
    )

    return tokenizer


@st.cache(allow_output_mutation=True)
def get_model() -> GPT2LMHeadModel:
    return AutoModelForCausalLM.from_pretrained(
        "bigcode/santacoder", revision="aaeed52", trust_remote_code=True
    ).to(device)


class TextSubstring(TypedDict):
    text: str
    start: int
    end: int


def preprocess_text(text) -> List[TextSubstring]:
    """Splits the text and returns a list of dicts containing the substring and
    its position in the original text. Currently just splits at the start and
    end of each whitespace-separated set of characters"""

    items = []
    for m in re.finditer(r"\S+", text):
        previtem = items[-1] if len(items) > 0 else None
        if previtem and previtem["end"] != m.start():
            items.append(
                {
                    "text": text[previtem["end"] : m.start()],
                    "start": previtem["end"],
                    "end": m.start(),
                }
            )
        items.append({"text": m.group(), "start": m.start(), "end": m.end()})

    return items


def generate_prompt(code: str, start: int, end: int) -> str:
    # print(f"Generating prompt for `{code[start:end]}`")
    # print("start", start, "end", end)

    max_prefix_length = 500
    max_suffix_length = 500

    prefix = code[max(0, start - max_prefix_length) : start]
    suffix = code[end : end + max_suffix_length]

    return FIM_PREFIX + prefix + FIM_SUFFIX + suffix + FIM_MIDDLE


@torch.no_grad()
def perform_inference(input_prompts, progress_callback=None):
    if len(input_prompts) == 0:
        return []

    outputs_per_input = 5

    tokenizer = get_tokenizer()

    inputs = tokenizer(
        input_prompts, return_tensors="pt", padding=True, return_token_type_ids=False
    ).to(device)

    outputs = get_model().generate(
        **inputs,
        # Only allow the model to generate up to 10 tokens. If we
        # need more than that to fix your code it's not a typo anymore!
        max_new_tokens=20,
        num_return_sequences=outputs_per_input,
        num_beams=outputs_per_input,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True,
        # Always force the model to generate the eos token. If an EOS is improbable, that's very important information!
        forced_eos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Adapted from https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175
    # Just the generated sequences, ignoring the prompt
    gen_sequences = outputs.sequences[:, inputs.input_ids.shape[-1] :]

    # For each sequence, get the probability of each token at each step
    probs = torch.stack(outputs.scores, dim=1).softmax(-1)

    gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)

    # Wherever the sequence is the pad token, set the probability to 1. That way we don't penalize short sequences. We already penalize premature truncation by forcing the model to generate the EOS token.
    gen_probs = torch.where(
        gen_sequences == tokenizer.pad_token_id,
        torch.ones_like(gen_probs),
        gen_probs,
    )

    # Multiply all the probabilities together to get the probability of the entire sequence being generated
    unique_prob_per_sequence = gen_probs.prod(-1)

    # Decode the returned sequences into strings
    sequences = tokenizer.batch_decode(gen_sequences, skip_special_tokens=True)

    sequences_with_probs = list(zip(sequences, unique_prob_per_sequence.tolist()))

    all_results = []

    for i in range(len(input_prompts)):
        batch_sequences = sequences_with_probs[
            i * outputs_per_input : (i + 1) * outputs_per_input
        ]

        batch_sequences = sorted(batch_sequences, key=lambda x: x[1], reverse=True)
        all_results.append(batch_sequences)
    return all_results


def format_output(processed_text: List[Dict[str, Any]]) -> str:
    css = """
    * {
        box-sizing: border-box;
    }
    
    html {
        background-color: #333;
    }
    
    html, body, #output {
        height: 100%;
    }
    
    #output {
      font-family: monospace;
      white-space: pre;
      padding: 20px;
      padding-bottom: 140px;
      min-height: 600px;
    }

    .token {
      color: white;
      position: relative;
      cursor: pointer;
    }

    .alternatives {
      position: absolute;
      display: none;
    }

    .token:hover {
      background-color: #ddd;
    }

    .token:hover .alternatives {
      color: #000;
      position: absolute;
      top: 14px;
      left: 0;
      display: block;
      background-color: #fff;
      z-index: 1;
      padding: 8px;
      border-radius: 8px;
      border: 1px solid #ccc;
    }
    
    tr {
        background-color: #eee;
    }
    
    tr:nth-child(even) {
      background-color: #fff;
    }
    """

    spans = []
    for entry in processed_text:
        token = entry["text"]
        suggestion = entry["output"] if "output" in entry else None

        if suggestion is None:
            spans.append(f"<span class='token'>{token}</span>")
            continue

        # Find the token in the suggestions list and determine how likely it is
        token_probability = 0
        for s, p in suggestion:
            if s == token:
                token_probability = p
                break

        top_probability = suggestion[0][1]

        token_color = "white"
        if top_probability > 0.5 and token_probability < 0.1:
            token_color = "red"

        suggestions_items = []
        for s, p in suggestion:
            if s == token:
                s = f"<strong>{s}</strong>"
            if s == "":
                s = "<i>remove token</i>"

            suggestions_items.append(f"<tr><td>{s}</td><td>{p:.2f}</td></li>")

        suggestions_table = f"<table><tr><th>Suggestion</th><th>Confidence</th>{''.join(suggestions_items)}</table>"

        spans.append(
            f"<span class='token' style='color: {token_color}'>{token}<span class='alternatives'>{suggestions_table}</span></span>"
        )

    return f"<style>{css}</style><body><div id='output'>{''.join(spans)}</div></body>"


def get_suggestions(text, progress):
    processed_text = preprocess_text(text)

    input_prompts = []
    prompt_index_to_processed_text_index = []

    for i, processed in enumerate(processed_text):
        substring = processed["text"]
        if i == 0:
            # don't generate suggestions for the first token since it's common for
            # the model to want to throw in extra lines at the beginning
            continue

        if substring.strip() == "":
            # Don't generate suggestions for whitespace
            continue

        input_prompts.append(
            generate_prompt(text, processed["start"], processed["end"])
        )
        if substring == '"":':
            print(input_prompts[-1])
        prompt_index_to_processed_text_index.append(i)

    # Chunk the prompts into batches of 10
    for i in range(0, len(input_prompts), 10):
        batch_prompts = input_prompts[i : i + 10]
        batch_prompt_index_to_processed_text_index = (
            prompt_index_to_processed_text_index[i : i + 10]
        )

        batch_outputs = perform_inference(batch_prompts)

        for token_index, output in zip(
            batch_prompt_index_to_processed_text_index, batch_outputs
        ):
            processed_text[token_index]["output"] = output

        progress.progress(min((i + 10) / len(input_prompts), 1.0))

    return format_output(processed_text)


st.set_page_config(layout="wide")

st.title("Semantic Linter")
st.text("Use the SantaCoder model as a linter to find potential typos.")

col1, col2 = st.columns(2)

with col1:
    code = st.text_area(
        label="Paste the code you'd like to lint here", value=sample_code, height=600
    )
    st.button("Lint Code")

with col2:
    st.text("Linter Output")
    progress = st.progress(0)
    components.html(get_suggestions(code, progress), height=600)

# if st.button("Generate Suggestions"):
#   # Print the output of get_suggestions(code)
