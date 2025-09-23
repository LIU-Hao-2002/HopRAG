# HopRAG: Multi-hop Reasoning for Logic-Aware Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for **HopRAG: Multi-hop Reasoning for Logic-Aware Retrieval-Augmented Generation**, accepted to ACL Findings 2025.

HopRAG is a novel Retrieval-Augmented Generation (RAG) framework that leverages graph databases to enhance multi-hop reasoning. Instead of treating documents as a flat collection, HopRAG models them as a graph of interconnected text chunks (nodes) within a **Neo4j** database. This structure allows for more sophisticated, logic-aware retrieval paths, enabling Large Language Models (LLMs) to answer complex questions that require synthesizing information from multiple sources.

We provide demonstration datasets from **HotpotQA**, **2WikiMultiHop**, and **MuSiQue** to get you started quickly.

-----

## 🚀 Getting Started

Follow these steps to set up the HopRAG environment and prepare for your first run.

### Prerequisites

  * Python `3.10.10` or later
  * **Neo4j Community Edition** `5.26.0` installed and running locally.

### Installation and Configuration

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/HopRAG-Lab/HopRAG.git
    cd HopRAG
    ```

2.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the Environment (`config.py`):**
    Before running any scripts, you must update `config.py` with your local setup details.

      * **Neo4j Connection:** Set your database credentials.

          * `neo4j_url`
          * `neo4j_user`
          * `neo4j_password`
          * `neo4j_dbname`

      * **LLM API:** Provide your API endpoint and key for generation.

          * `personal_base`
          * `personal_key`
          * `default_gpt_model`

      * **Embedding Model:** Specify the path to your locally downloaded embedding model. This model must be the same for both building the graph and retrieval.

          * `embed_model`
          * `embed_model_dict`
          * `embed_dim`

      * **(Optional) Local Models:** If you are using local models for generation or reranking, update their paths.

          * `reranker`
          * `query_generator_model`
          * `traversal_model`

-----

## ⚙️ Usage: A Step-by-Step Guide

Follow this pipeline to build the graph, run retrieval, and generate answers.

### Step 1: Prepare the Dataset

First, preprocess your dataset (`.json` or `.jsonl`) using `data_preprocess.py`. This script converts the data into a standardized `.jsonl` format and extracts all document passages into a directory of `.txt` files (the "doc pool").

  * For **HotpotQA** or **2WikiMultiHop**, use the `main_hotpot_2wiki` function.
  * For **MuSiQue**, use the `main_musique` function.

**Note:** Ensure the sentence delimiter used in this step (e.g., `\n\n`) matches the `signal` variable in `config.py` for consistent document chunking.

### Step 2: Build Graph Nodes

This step chunks the documents from your doc pool and creates a node for each chunk in the Neo4j database. Run the `main_nodes` function in `HopBuilder.py`.

**Key Parameters:**

  * `docs_dir`: Path to the doc pool directory created in Step 1 (e.g., `quickstart_dataset/hotpot_example_docs`).
  * `cache_dir`: A directory to log progress. This allows the script to be resumed after an interruption.
  * `node_name`: A unique name (type) for your nodes in Neo4j (e.g., `hotpot_bgeen_qwen1b5`). Set this in `config.py`.

We recommend the **separate offline-online mode** for faster and more stable node creation.

#### Mode 1: Separate (Recommended)

1.  **Generate nodes offline:** This step processes the documents and saves the node data locally without connecting to Neo4j.
    ```python
    # In HopBuilder.py
    main_nodes(cache_dir='quickstart_dataset/cache_hotpot_offline',
               docs_dir="quickstart_dataset/hotpot_example_docs",
               label=node_name)
    ```
2.  **Push nodes to Neo4j:** This step uploads the locally cached nodes to your online database.
    ```python
    # In HopBuilder.py
    main_nodes(cache_dir='quickstart_dataset/cache_hotpot_online',
               docs_dir="quickstart_dataset/hotpot_example_docs",
               label=node_name,
               original_cache_dir='quickstart_dataset/cache_hotpot_offline')
    ```

#### Mode 2: Hybrid (Alternative)

This mode processes and uploads nodes in a single step.

```python
# In HopBuilder.py
main_nodes(cache_dir='quickstart_dataset/cache_hotpot_online',
           docs_dir="quickstart_dataset/hotpot_example_docs",
           label=node_name,
           offline=False)
```

### Step 3: Build Edges and Index

Next, connect the nodes with edges and create the vector and keyword indices needed for efficient retrieval. Run the `main_edges_index` function in `HopBuilder.py`.

  * **Specify Index Names:** Before running, define your index names in `config.py`:
      * `node_dense_index_name`
      * `edge_dense_index_name`
      * `node_sparse_index_name`
      * `edge_sparse_index_name`
  * **Run the script:** The `main_edges_index` function uses dataset-specific logic (e.g., `create_edges_hotpot` or `create_edges_musique`) to create edges based on the ground truth paths in your training data.

After this step, your graph is fully built and indexed, ready for retrieval\!

### Step 4: Test Retrieval (Optional)

To verify that the graph and retrieval functions are working correctly, you can run a standalone search using the `search_docs` function in `HopRetriever.py`. This is a great way to debug or experiment with different retrieval hyperparameters (`max_hop`, `topk`, `traversal`, etc.) before running the full generation pipeline.

### Step 5: Retrieval-Augmented Generation

Now, run the end-to-end RAG pipeline using `HopGenerator.py` from your command line. This script retrieves relevant context from the graph and passes it to the LLM to generate the final answer.

**Example Command:**

```bash
nohup python3 HopGenerator.py \
    --model_name 'gpt-3.5-turbo' \
    --data_path 'quickstart_dataset/hotpot_example.jsonl' \
    --save_dir 'quickstart_dataset/hotpot_output' \
    --retriever_name 'HopRetriever' \
    --max_hop 4 \
    --topk 20 \
    --traversal 'bfs' \
    --mode 'common' \
    --label 'hotpot_bgeen_qwen1b5' > hotpot_run_log.txt &
```

The script will generate a results file formatted for official evaluation scripts and a `cache` directory with detailed logs for each question.

### Step 6: Evaluation

The output files produced in the previous step are ready for evaluation. Use the corresponding official evaluation tools for your benchmark (e.g., the HotpotQA evaluation suite) to measure performance.

-----

## 📜 Citing HopRAG

If you find our work useful in your research, please cite our paper:

```bibtex
@article{liu2025hoprag,
  title={{HopRAG}: Multi-hop reasoning for logic-aware retrieval-augmented generation},
  author={Liu, Hao and Wang, Zhengren and Chen, Xi and Li, Zhiyu and Xiong, Feiyu and Yu, Qinhan and Zhang, Wentao},
  journal={arXiv preprint arXiv:2502.12442},
  year={2025}
}
```

Thank you for your interest in HopRAG\!
