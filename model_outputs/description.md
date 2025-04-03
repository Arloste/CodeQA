# This folder contains the output of the model with different parameters, and also evaluation results

### Model outputs
TODO remove the prompt from questions and keep the user question only

1. `Incompelete graph & RAG LLM.json` - simple RAG. No .txt files included
2. `Incomplete graph & Graph LLM.jsonl` - Graph LLM. No .txt files included

3. `Full graph & RAG LLM.jsonl` - simple RAG wihout GNNs. Graph structure extended with text files and updated node information
4. `Full graph & GNN.txt` - Graph LLM. Extended graph structure

### Evaluations on model vs model

All evaluations of two models against each other are run twice: on the first run, the first model's answers are written first; and on the second run, the first model's ansewrs are written latter. This approach mitigates the position bias, that may be introduced by LLM-as-a-judge.

`inc_rag_vs_graph.jsonl`, `inc_graph_vs_rag.jsonl` - Comparing RAG and Graph LLM on an incomplete graph.
`full_rag_vs_graph.jsonl`, `full_graph_vs_rag.jsonl` - Comparing RAG and Graph LLM on a full graph.

### Evaluations on one model

`abs_incomp_rag.jsonl` - shows the quality of RAG (output #1) against ground truth answers on an incomplete files.
`abs_incomp_graph.jsonl` - shows the quality of Graph LLM (output #2) against ground truth answers on an incomplete files.
`abs_full_rag.jsonl` - shows the quality of RAG (output #3) against ground truth answers on a full graph.
`abs_full_graph.jsonl` - shows the quality of Graph LLM (output #4) against ground truth answers on a full graph.