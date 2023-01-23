# Typosaurus

Typosaurus is a semantic linter that automatically detects typos and errors in your code. It is built using SantaCoder LLM, a 1.1B parameter model (https://huggingface.co/bigcode/santacoder) and uses its Fill-in-the-middle mode to detect errors that normal linters are not able to pick up. It is currently just a proof of concept.

## How It Works

Typosaurus "blanks out" a substring from the code sample and gives SantaCoder the text before/after it. It then asks SantaCoder to predict the missing substring. If SantaCoder has a high-confidence prediction that doesn't match the actual code in the file, Typosaurus flags it as a potential bug.

For example, if the code snippet is:

```
def hello():
  print("hello world")
```

Typosaurus would split the code into substrings (currently naively based on whitespace) as `['def', 'hello():', 'print("hello', 'world")']`. It then rearranges the context around each substring into a prompt in a special [fill-in-the-middle format](https://huggingface.co/bigcode/santacoder#fill-in-the-middle) SantaCoder was trained on.

For example, for the `hello():` substring above it would create a prompt like this

```
<fim-prefix>def <fim-suffix>\n    print("Hello world!")<fim-middle>
```

SantaCoder generates candidate completions for the missing substring. We then look at whether any of the strings SantaCoder generated match the string from the actual source we're evaluating. If they're different and SantaCoder has high confidence in its generation, we flag it as a potential bug.

## Example Output

This is an example of the [Streamlit demo](./streamlit-demo.py) when passed in a chunk of code with an error.

https://user-images.githubusercontent.com/176426/213827814-c9099877-e821-49db-a299-b6a444eb8577.mov

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
