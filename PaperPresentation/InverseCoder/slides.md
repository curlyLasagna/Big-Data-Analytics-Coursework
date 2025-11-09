---
theme: apple-basic
title: Inverse Coder
---

<div class="flex flex-col items-center justify-center">
<h1>Inverse Coder</h1>
    <p class="text-center w-100">
        Generating instruction-tuned dataset using the intelligence of open-source code models instead of Big Tech.
    </p>
  <img src="/logo.png" class="mb-4 w-32 h-32 object-cover" />
</div>

---

## Overview

### Inverse-Instruct
- Novel approach of generating new instruction dataset to train models

### InverseCoder
- LLMs trained using Inverse-Instruct
- How it stand against the competition

---

## Background

### Instruction Tuning LLMs

An LLM without instruction tuning might respond to a prompt of “_teach me how to bake bread_” with “_in a home oven.”_
- This is due to the nature of LLMs predicting the next best token. LLMs only append text.

Achieved by fine-tuning a pre-trained model with a dataset composed of instructions and the supposed outputs.

### Motivation

- Obtaining high quality instruction set is difficult and expensive
- Prior, open sourced data generation techniques rely on querying powerful closed source models from OpenAI to get more training data

---

## Inverse-Instruct

### Code Preprocessing

Filter LLM responses that contain markdown code block syntax, which are the triple tick marks
```ts {1|3}
👉```
<Code snippet\>
👉```
```

i.e.

> ```python
> def foo():
>   pass
> ```
> Function `foo` does this

---

### Code Preprocessing

Filter out the natural language surrounding the code snippets.

LLMs responses often show examples of how to use the snippet, whether they be functions or classes, so we disregard those and keep the first part.

```json{*}{maxHeight:'350px'}
[
  {
  "Instruction": Create a function to search for a word in an array. The word should be searched in the lowercase version of the array and it should return the index location of its first occurrence.
  word = "test"
  words_array = ["Hello", "World", "Test", "Test", "me"],
  "Response": Here's how you can create the function in Python:

  \```python
  def search_word(word, words_array):
  word = word.lower()
  words_array = [w.lower() for w in words_array]
  if word in words_array:
  return words_array.index(word)
  else:
  return "Word not found"

  word = "test"
  words_array = ["Hello", "World", "Test", "Test", "me"]
  print(search_word(word, words_array))
  \```
  This script turns both the search term and the words in the array to lower-case to ensure the search is case-insensitive. It checks if the word is within the array. If it is, it returns the index of the first occurrence of the word. If not, it returns a "Word not found" message.
  }
]

```

---

````md magic-move
```
Here's how you can create the function in Python:

  ```python
  def search_word(word, words_array):
  word = word.lower()
  words_array = [w.lower() for w in words_array]
  if word in words_array:
    return words_array.index(word)
  else:
    return "Word not found"

  word = "test"
  words_array = ["Hello", "World", "Test", "Test", "me"]
  print(search_word(word, words_array))
  ```

  This script turns both the search term and the words in the array to lower-case to ensure the search is case-insensitive. It checks if the word is within the array. If it is, it returns the index of the first occurrence of the word. If not, it returns a "Word not found" message.
```

```python
  def search_word(word, words_array):
    word = word.lower()
    words_array = [w.lower() for w in words_array]
    if word in words_array:
      return words_array.index(word)
    else:
      return "Word not found"

  word = "test"
  words_array = ["Hello", "World", "Test", "Test", "me"]
  print(search_word(word, words_array))
```

````

- $y_i$: The original trained data
- $y^*_i$: Preprocessed code snippet

---

### Code Summarization

Prompts an instruction-tuned LLM (WizardCoder-GPT4) to summarize each pre-processed code snippets alongside multiple new instructions

The prompt provided by the author to the summarizing LLM:
```
@@ Instruction
This is a response code snippet to a programming problem, please give the problem description:

@@ Response
Write a / Create a /
Implement a / Develop a /
Design a / Build a / I want a

```

---

| Generation Method                                   | WC-CL-7B | WC-DS-6.7B |
|-----------------------------------------------------|----------|------------|
| NL → Code                                           | 62.4     | 70.2       |
| Code → NL →<sub>GPT-4</sub> Code                    | **74.3** | **79.0**   |
| Code → NL →<sub>Humans</sub> Code                   | **86.7** | **80.0**   |

Evaluated with MBPP via Pass@k metrics

_Will elaborate the above evaluation on later slides_

- `Code -> NL -> GPT-4`: WC-CL-7B generated the natural language from code. GPT-4 reads in the natural language and responds with code
- `Code -> NL -> Human`: WC-CL-7B generated the natural language from code. Human reads in the natural language and responds with code

---

Code Snippet ($y^*_i$)

```python
def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
```

The LLM could return the following instructions ($x^*_{ij}$)


```json
[
"Find the indices of two numbers in the list whose sum is equal to the target value.",
"Return a list containing the indices of two different elements whose sum equals the target.",
"Given an array and a target number, identify two elements such that their addition results in the target and return their indices."
]
```

This results in the pairing of $\{(x^*_{ij},\; y^*_i)\}$
---

### Self Evaluation and Data Selection

A few of the entries of the summary generated by WizardCoder-GPT4 $(x^*_{ij})$ could be mistakes, so we'll need a process of selecting the best instruction ($x^{**}_i$)

This step results should return this pair: $\{(x^{**}_i, y^{*}_i)\}$

> **Prompt**:

> @@ Instruction

> This is a response code snippet to a programming problem,

> please give the problem description:

> Here is a programming problem:

> {instruction}

> Here is the answer code to the problem:

> {code}

> Is the answer correct? Your reply should begin with Yes or No.

---

Output (Simplified):

| Instruction | Model Response | Logprob("YES") | Logprob("NO") |
|---------|---------------|----------------|---------------|
| "Yes, foo"       | YES           |  -0.05         |   -3.0        |
| "No, bar"       | NO            |  -2.3          |   -0.12       |
| "no, baz"       | NO            |  -1.9          |   -0.4        |

---

#### Pseudo-Probability of Yes

How the best instruction from a list of instructions is selected.

The paper defines Pseudo-Probability of Yes formula as:

$$
\text{LM-Score}( \cdot ) = \frac{\text{exp}(logit(Yes))}{\text{exp}(logit(Yes)) + \text{exp}(logit(No))}
$$

or in the Python source code

```python
# np.exp is $e^x$
np.exp(yes_logprob) / (np.exp(yes_logprob) + np.exp(no_logprob))
```

Where we select the instruction with the highest `LM-Score`

---

###### Probability for each token

LLM's are autocomplete on steroids or "token prediction machine"

<img src="/prob.png" class="h-auto max-w-xs center mx-auto">

---

###### Softmax

OpenAI's logprob values range: $(-\infty, 0]$

where $0$ corresponds to 100% probability

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum^K_{j=1}e^{z_j}}
$$

Find the softmax of the logprob to get the probability that sums to 1

(Basically undo the logarithm of log prob)

---

###### Why not just return the probability?

It's more efficient to add the values of log probabilities than multiplying small probability values.

$$P(\text{Sequence}) = P(\text{word}_1) \times P(\text{word}_2 | \text{word}_1) \times \dots \times P(\text{word}_N | \text{context})$$


$$\log(P(\text{Sequence})) = \log(P(\text{word}_1)) + \log(P(\text{word}_2 | \text{word}_1)) + \dots$$

Remember: $\log(p * q) = \log(p) + \log(q)$

The product could also become very small, it's rounded to 0, losing information

---

<div class="grid grid-cols-2 gap-4">
  <div class="p-4">
Let

$z_{\mathrm{yes}} = \text{log-probability of "yes"}$

$z_{\mathrm{no}} = \text{log-probability of "no"}$
  </div>
  <div class="p-4">
$$
P(\mathrm{yes}) = \frac{e^{z_{\mathrm{yes}}}}{e^{z_{\mathrm{yes}}} + e^{z_{\mathrm{no}}}}
$$
  </div>
</div>

Example

<div class="grid grid-cols-2 gap-4">
  <div class="bg-blue-100 p-4">

$z_{\text{yes}} = -0.4$

$z_{\text{no}} = -1.8$

  </div>
  <div class="bg-green-100 p-4">
$$
\begin{aligned}
  P(\mathrm{yes}) & = \frac{e^{-0.4}}{ e^{-0.4} + e^{-1.8} } \\
  & = \frac{0.6703}{0.6703 + 0.1653} \\
  & = \frac{0.6703}{0.8356} \\
  & \approx 0.803 \\
\end{aligned}
$$
  </div>
</div>

---

#### Summary

![](/overview.png)

---

## Training

### Base Models
- CodeLlama-Python-13B
- CodeLlama-Python-7B
- DeepSeek-Coder-Base-6.7B

Trained with the new dataset generated by InverseInstruct and codealpaca-v1 instruction and response dataset

---

### Models to compare

- MagicoderS
- WaveCoder-Ultra-DS
- AlchemistCoder
- WizardCoder-GPT4

These competing models are trained on the original training dataset from codealpaca-v1 and GPT-3.5 generated training data with their respective techniques.

---

### How competing models train

#### MagicoderS

Uses OSS-instruct to generate data. Prompts a powerful closed source LLM with a random open source code snippet to generate a problem that describes the code snippet and to generate a code solution to the problem it just created

![](/oss_instruct.png)


---

#### WizardCoder

Uses evol-instruct to generate data. Prompts a powerful close source LLM to mutate or create a new instruction from the initial instruction into something more complex for several rounds.

Take the evolved instructions to generate code from it.

<div class="flex justify-center">
<img src="/evol.webp" style="height:350px">
</div>

---

#### WaveCoder

A model trained on CodeSeaXDataset, an instruction dataset covering 4 code-related tasks:
- Generation
- Summarization
- Translation
- Code fix

- Reads in open source code snippets and prompts a powerful close source model (GPT-4) for instructions bin that code snippet on any one of the above tasks based on content via a task definition metadata.
- GPT-3.5 either generates an instruction and code response pair or summarize code and attaches that summary as instruction
- GPT-4 is prompted again as quality control on the dataset generated


---

#### AlchemistCoder

Uses AlchemistPrompts to generate new data.

Using a powerful close source LLM:

- Take source code from multiple sources and transforms it into a uniform `instruction` and `output`.
- Rewords instructions by clarifying intent, add reasoning, and consistent structure
- Filter out mismatched instruction-output pairs

<img src="/alchemist.png" style="height:250px">

---

### Benchmarks

- HumanEval:
  - 164 Python programming problem dataset by OpenAI

- MBPP (Mostly Basic Python Problems):
  - 1,000 crowd-sourced Python programming problems
  - Subset of the data are verified by humans

- DS-1000:
  - 1,000 data science problems that spans across popular Python libraries such as Numpy and Pandas

- MultiPL-E:
  - Took Python benchmarks and compiled it to 18 other programming languages to keep it consistent

---

All benchmarks are measured via `pass@k` metric
  - Measures the probability that at least one of the top k generated code samples for a problem passes a single unit test

---

## Results

Inverse-Instruct showcases the following improvements to a base model:
- General Python generation capabilities
- Improvements across different programming languages
- Improvements in data science code generation tasks

---

| Model | HumanEval (+) | MBPP (+) |
|-------|----------------|-----------|
| GPT-4-Turbo (April 2024) | **90.2 (86.6)** | **85.7 (73.3)** |
| GPT-3.5-Turbo (Nov 2023) | <u>76.8</u> (<u>70.7</u>) | <u>82.5</u> (<u>69.7</u>) |
| ***Based on CodeLlama-Python-13B*** |  |  |
| CodeLlama-Python-13B | 42.7 (38.4) | 63.5 (52.6) |
| WizardCoder-GPT4-CL-13B | <u>76.8</u> (<u>70.7</u>) | <u>73.5</u> (<u>62.2</u>) |
| **InverseCoder-CL-13B** | **79.9 (74.4)** | **74.6 (63.0)** |

---

| Model | HumanEval (+) | MBPP (+) |
|-------|----------------|-----------|
| GPT-4-Turbo (April 2024) | **90.2 (86.6)** | **85.7 (73.3)** |
| GPT-3.5-Turbo (Nov 2023) | <u>76.8</u> (<u>70.7</u>) | <u>82.5</u> (<u>69.7</u>) |
| ***Based on CodeLlama-Python-7B*** |  |  |
| CodeLlama-Python-7B | 37.8 (35.4) | 59.5 (46.8) |
| Magicoder-S-CL-7B | 70.7 (67.7) | **70.6 (60.1)** |
| AlchemistCoder-CL-7B | 74.4 (68.3) | 68.5 (55.1) |
| WizardCoder-GPT4-CL-7B | <u>72.6</u> (<u>68.9</u>) | <u>69.3</u> (<u>59.3</u>) |
| **InverseCoder-CL-7B** | **76.2 (72.0)** | **70.6 (60.1)** |

---

| Model | HumanEval (+) | MBPP (+) |
|-------|----------------|-----------|
| GPT-4-Turbo (April 2024) | **90.2 (86.6)** | **85.7 (73.3)** |
| GPT-3.5-Turbo (Nov 2023) | <u>76.8</u> (<u>70.7</u>) | <u>82.5</u> (<u>69.7</u>) |
| ***Based on DeepSeek-Coder-6.7B*** |  |  |
| DeepSeek-Coder-6.7B | 47.6 (39.6) | 72.0 (58.7) |
| Magicoder-S-DS-6.7B | 76.8 (71.3) | **79.4** (**69.0**) |
| WaveCoder-Ultra-DS-6.7B | 75.0 (69.5) | 74.9 (63.5) |
| AlchemistCoder-DS-6.7B | **79.9** (75.6) | 77.0 (60.2) |
| WizardCoder-GPT4-DS-6.7B | <u>77.4</u> (<u>73.2</u>) | 77.8 (<u>67.5</u>) |
| **InverseCoder-DS-6.7B** | **79.9 (76.8)** | <u>78.6</u> (**69.0**) |

---

### DS-1000

| Model                  | plt.   | np.    | pd.    | torch  | scipy  | sklearn | tf.    | All    |
|------------------------|--------|--------|--------|--------|--------|---------|--------|--------|
| *(Based on DeepSeek-Coder-6.7B)* |        |        |        |        |        |         |        |        |
| **magicoders**         | _54.8_ | _48.9_ | _30.0_ | 49.2   | 27.3   | 44.7    | _41.2_ | 41.2   |
| **WizardCoder-GPT4**   | 53.8   | **53.9** | 28.0   | _49.3_ | **30.4** | _45.7_  | **44.4** | _42.2_ |
| **InverseCoder**| **55.5** | **53.9** | **32.3** | **56.7** | _30.0_ | **50.3** | 33.9   | **44.2** |

---

| Model                  | plt.   | np.    | pd.    | torch  | scipy  | sklearn | tf.    | All    |
|------------------------|--------|--------|--------|--------|--------|---------|--------|--------|
| *(Based on codellamapy-13B)* |        |        |        |        |        |         |        |        |
| **WizardCoder-GPT4**   | **56.1** | _52.2_ | _30.3_ | _43.0_ | **25.2** | _49.5_  | _40.0_ | _42.1_ |
| **InverseCoder**| _53.0_ | **54.3** | **32.1** | **50.9** | _22.5_ | **50.5** | **43.8** | **43.1** |

---

| Model                  | plt.   | np.    | pd.    | torch  | scipy  | sklearn | tf.    | All    |
|------------------------|--------|--------|--------|--------|--------|---------|--------|--------|
| *(Based on codellamapy-7B)* |        |        |        |        |        |         |        |        |
| CodeLlama-Python       | _55.3_ | 34.5   | 16.4   | 19.9   | 22.3   | 17.6    | 28.5   | 28.0   |
| WizardCoder            | 53.5   | 34.4   | 15.2   | 25.7   | 21.0   | 24.5    | 28.9   | 28.4   |
| **magicoders**         | **55.9** | 40.6   | _28.4_ | _40.4_ | 28.8   | _35.8_  | 37.6   | 37.5   |
| **WizardCoder-GPT4**   | 51.5   | _46.9_ | **29.9** | **43.6** | **34.9** | **41.9** | _39.0_ | **40.2** |
| **InverseCoder**| 54.2   | **48.6** | 27.4   | 38.0   | _34.0_ | **41.9** | **40.3** | _39.9_ |

---

### MultiPL-E


| Model                   | Java   | JS     | C++    | PHP    | Swift  | Rust   | Avg.   |
|-------------------------|--------|--------|--------|--------|--------|--------|--------|
| *(Based on CodeLlama-Python-13B)* |        |        |        |        |        |        |        |
| **WizardCoder-GPT4***   | **55.4** | _64.2_ | _55.9_ | _52.0_ | _49.9_ | _53.4_ | _55.1_ |
| **InverseCoder***| 54.5   | **65.4** | **58.1** | **55.3** | **52.5** | **55.6** | **56.9** |
| *(Based on CodeLlama-Python-7B)* |        |        |        |        |        |        |        |
| CodeLlama-Python        | 29.1   | 35.7   | 30.2   | 29.0   | 27.1   | 27.0   | 29.7   |
| **magicoders***         | _49.8_ | **62.6** | 50.2   | _53.3_ | 44.9   | 43.8   | 50.8   |
| **WizardCoder-GPT4***   | **50.4** | 60.7   | _50.6_ | 51.6   | _45.6_ | **48.2** | _51.2_ |
| **InverseCoder***| 48.7   | _61.9_ | **52.6** | **55.2** | **53.0** | _46.1_ | **52.9** |

---

| Model                   | Java   | JS     | C++    | PHP    | Swift  | Rust   | Avg.   |
|-------------------------|--------|--------|--------|--------|--------|--------|--------|
| *(Based on DeepSeek-Coder-6.7B)* |        |        |        |        |        |        |        |
| **magicoders***         | 59.6   | _69.8_ | _70.0_ | **64.4** | **54.4** | 53.6   | _62.0_ |
| **WizardCoder-GPT4***   | **61.4** | 66.4   | 68.7   | 61.8   | 52.6   | _56.1_ | 61.2   |
| **InverseCoder***| _60.7_ | **70.1** | **70.5** | _63.6_ | _53.0_ | **57.4** | **62.6** |
---

### Limitations

Training Inverse-Instruct generated code to an InverseCoder LLM results in worsening performance due to model collapse, causing less diverse responses.

| Model                  | HumanEval (+)        | MBPP (+)           |
|------------------------|----------------------|--------------------|
| **InverseCoder-CL-7B** | **76.2** (**72.0**)  | **70.6** (_60.1_)  |
| _InverseCoder-CL-7B-V2_| _75.0_ (_70.1_)      | **70.6** (**60.6**)|

---

<div class="flex justify-center item-center">
<h1>Questions?</h1>
</div>
