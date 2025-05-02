# This folder contains the output of the model with different parameters, and also evaluation results

### Model outputs

Model predictions on the Spyder IDE CodeQA dataset.

Each file corresponds to a different run with different parameters. For example, `G-Retriever 1024 items.jsonl` contains the output of G-Retriever with the context length of 1024 tokens.

Each line in the files corresponds to a different datapoint from the dataset in the json format. It includes the question, correct answer, predicted answer, and the question ID.


### Model vs Model evaluation

Original evaluation approach for SpyderIDE CodeQA Benchmark.

This approach compares the output of two models, as compared with the question and the correct answer to the question. There are four possible scores: `Assistant's A answer is better`, `Assistant B's answer is better`, `Both assistants are good` (if both models have shown similar answers that match the correct answer), and `Both assistants are bad` (if both model have answered incorrectly).

All evaluations of this type are run twice: on the first run, the first model's answers are written first; and on the second run, the first model's answers are written latter. This approach mitigates the position bias, that may be introduced by the LLM-as-a-judge. The final conclusion is the average number of good results of each pair of evaluations.

The evaluations include:
 - #01 - #14: Comparison between RAG and G-Retriever for different context length (2, 4, 16, 128, 512, 1024, 2048)
 - #15 - #18: Comparison between RAG and fine-tuned G-Retriever for different context length (128 and 512).
 - #19 - #22: Comparison between the baseline G-Retriever and the fine-tuned G-Retriever for different context length (128 and 512).
 - #23 - #24: Comparison between the context lengths of 128 and 512 of the fine-tuned G-Retriever.


### One Model Evaluation

The evaluation approach included by me. In this approach, the LLM-as-a-judge evaluates the answer on the scale from 0 to 10 based on how correct and helpful it is for answering the user question. Since the main application for this system is to answer the user questions in real life, this metric is more important to measure how useful the assistant is.

Due to the fact, that LLM-as-a-judge may not give the same score to the same question, each evaluation is conducted three times. The final score is calculated as the median score of these three scores.

### Scripts

All evaluations were conducted in Google Colab. Models and [code](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb) from the [Unsloth team](https://huggingface.co/unsloth) were used.

The Scripts folder contains the notebooks, along with the prompts, that were used to make Model vs Model and One Model evaluations.