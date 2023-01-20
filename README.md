# Typosaurus

Typosaurus is a semantic linter that automatically detects typos and errors in your code. It is built using SantaCoder LLM, a 1.1B parameter model (https://huggingface.co/bigcode/santacoder) and uses its Fill-in-the-middle mode to detect errors that normal linters are not able to pick up.

## How It Works

Typosaurus extracts a substring from the code sample and gives the LLM the strings before/after it. It then asks SantaCoder to predict the string in the middle. If SantaCoder has a high-confidence prediction that doesn't match the actual code in the file, Typosaurus flags it as a potential bug.

For example, if the code snippet is:

```
def hello():
  print("hello world")
```

Typosaurus would split into substrings (currently naively based on whitespace) eg ['def', 'hello():', 'print("hello', 'world")']. It would then for each substring rearrange it into a prompt in a special fill-in-the-middle format SantaCoder was trained on https://huggingface.co/bigcode/santacoder#fill-in-the-middle

For example, for the `hello():` substring it would create a prompt like this

```
<fim-prefix>def <fim-suffix>\n    print("Hello world!")<fim-middle>
```

SantaCoder is trained to generate the code in the middle after the special <fim-middle> substring. We then look at whether the code SantaCoder generates matches the snippet from the actual source we're evaluating. If they're different and SantaCoder has high confidence in its generation, we flag it as a potential bug.

## Files

Typosaurus includes the following files:

- `explore.ipynb`: playing around with the SantaCoder model
- `streamlit-demo.py`: When run using Streamlit, lets you paste in a block of code and scan it for errors.

## Running

To run Typosaurus you'll need a machine with at least 10GB of VRAM.

Steps:

- Install mamba (https://mamba.readthedocs.io/en/latest/installation.html)
- Install the project dependencies: `mamba env update --prune -f environment.yml`
- Activate the environment: `conda activate typosaurus`
- Run the demo: `streamlit run streamlit-demo.py`

## Note

Typosaurus is currently just a proof of concept.
