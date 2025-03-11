# Introduction  
HopRAG is a Graph-based RAG project that stores passages in a Neo4j graph database(version:community 5.26.0), builds an index, and performs retrieval-augmented generation. It can be used for specific RAG evaluation datasets. 
Note:All paths in the project files need to be replaced with your local paths. We provide demo datasets from HotpotQA and MuSiQue for quickstart in `quickstart_dataset`

## 0. Neo4j, LLM api, embedding model Preparation
- Make sure you download neo4j database locally and can log into your account freely. Change the `neo4j_url`,`neo4j_user`,`neo4j_password`,`neo4j_dbname` in `config.py` so that you can log into it through python.
- Change the `personal_base`,`personal_key`,`default_gpt_model` in `config.py` so that you can call LLM through api in python.
- Make sure you download the embedding model locally and change the `embed_model`,`embed_model_dict`,`embed_dim` in `config.py`.

## 1. Prepare the Dataset
Preprocess `.json` format test set file using the `process_data` function in the `data_preprocess.py` file. The preprocessing will write the passages for all the questions from the test set as `.txt` files into the specified directory and transform the `.json` test file into `.jsonl` format. You might need to revise it according to the format of the specific dataset. 
- for hotpotqa or 2wiki dataset, you can proprocess it using `main_hotpot_2wiki` function. Please note that here the function will use `\n\n` to join sentences, which may affect your chunking delimeter `signal` in `config.py`. Please make sure they are consistent and customized to your chunking need. 
- for musique dataset, since it is already in `.jsonl` format, we can use `main_musique` function to write the doc pool.Since different questions from musique dataset may include the same doc for context, we also produce a id2txt `.json` file to create the mapping from each id to its unique doc text. 
We prepare two example datasets from hotpotqa and musique for quickstart, which are `quickstart_dataset/hotpot_example.json` and `quickstart_dataset/musique_example.json` respectively.


## 2. Build Nodes  
Run the `main_nodes` function in the `HopBuilder.py` file to build nodes in the graph. Please note that the following variable needs to be reset before you begin:
- Each node contains a chunk from a document text file in the doc pool directory (for example `quickstart_dataset/hotpot_example_docs` ). You can change the chunking delimeter by setting the `signal` in `config.py` according to the txt file in the doc pool.  
- In neo4j, each node,edge and index has a type(namely its name). You can change these three names by chaning the following variables:
    - for node type: `label` in `HopBuilder.py` ; 
    - for edge type: the substring in `create_pending2answerable` and `create_abstract2answerable` in `config.py`(here they are `pen2ans_hotpot_example` but you can change it according to your need)
    - for index type: the four `index` variables in `create_index` function in `HopBuilder.py`, respectively for node_dense_index, edge_dense_index, node_sparse_index, edge_sparse_index. The index name will be used during retrival so please make sure they are consistent with the index names in `HopRetriever.py`. Please also note that here the two `type` variables in `create_index` function (for demonstration they are `pen2ans_hotpot_example`) should also be consistent with the substring edge type in `create_pending2answerable` and `create_abstract2answerable` in `config.py`
- In `main_nodes` function, the following variables are introduced:
    - `cache_dir`: to save the nodes that are finished. Since building nodes from a large pool of documents may be a time-consuming job for large dataset and may encounter interruption, `HopBuilder.py` is designed to skip the document that has been processed by checking the `cache_dir` and continue from where it is interrupted. Here it is `quickstart_dataset/cache_hotpot` for demonstration.
    - `docs_dir`: the doc pool directory from step1. Here it is set as `quickstart_dataset/hotpot_example_docs` for demonstration.


## 3. Build Edges and Index
Run the `main_edges_index` function in the `HopBuilder.py` file.
In `main_edges_index` funtion:
- `cache_dir` should be the same the `cache_dir` used for `main_nodes` function. The `cache_dir` can cover the builder stage for one dataset(for example  HotpotQA) but should be re-initialized for another dataset (for example MuSiQue )
- `problems_path` should be the `.jsonl` file from stage1 containing the specific queries.
- Make sure you specify the four `index` and `type` varibales in `create_index` function since `create_index` will be called after finishing creating edges. 
- `create_edges_hotpot` function will create edges among nodes after finishing building nodes for the doc pool of HotpotQA and this function is tailored for the specific format of HotpotQA( and 2wiki since they basically share the same format). You may notice that there is a `create_edges_musique` function that highly resembles `create_edges_hotpot`. It is tailored for the format of MuSiQue. You can change the specific function used in `main_edges_index`. Here we demonstrate the version with `create_edges_hotpot`. But different datasets share the same `main_nodes` function. 

After finishing building nodes and edges, the `cache_dir` directory will contain:
- `docid2nodes.json` is a dict that maps from the local txt file name (str) to a list of node id(list of int) in the online database.  
- `node2questiondict.pkl` is a dict storing the mapping from the node in neo4j to its question dict. The specific format is the same as the `node2questiondict` in `create_nodes` function (Dict[Tuple[int,str],Dict[str,List[Tuple[str,Set,np.ndarray]]]]).
- `edges_done.pkl` is a pickle file for a set that contains all the local text files' names which have finished building edges. 
These three files can be used when incrementally updating the graph. For example when more questions or docs are added, these cache files will help `HopBuilder.py` avoid building new nodes for the same docs.
## 4. Retrieval
Once the graph-structure is built, you can try retrieving the context with a given query by the `search_docs` function in `HopRetriever.py`. But please note that this step is for testing the graph database and the retrieval phase, but not for generation. In step 5 the `HopGenerator.py` will use the `HopRetriever.py` to retrieve context and then generate response. For `HopRetriever.py`, there are some variables that need to be set properly according to your own need:
- `label` in `__init__`:the index prefix for each dataset in neo4j database. For example if the 4 index names in `HopBuilder.py`(`hotpot_example_node_dense_index`,`hotpot_example_edge_dense_index`,`hotpot_example_node_sparse_index`,`hotpot_example_edge_sparse_index`) shares the same prefix `hotpot_example_` then label should be set as `hotpot_example_`. By such design we can retrieve differnent contents from different index, for queries from different datasets by simply shifting the index prefix, even if the nodes from different datasets are stored in the same database in neo4j.
- `llm` in `__init__`: the llm used in retrieval for reasoning-augmented graph traversal ( not for generation).
- `embed_model` for `self.emb_model`. This should be the exact embedding model in `config.py`.
- other hyperparameters: `max_hop`,`entry_type`,`topk`,`traversal` etc. Please refer to the corresponding function for their usage.

We also present an example at the bottom of `HopRetriever.py`. If you get a list of str as `context`, feel free to continue!


## 5. Retrieval-augmented Generation  
Once the retrieval function is tested, run the `main` function in the `HopGenerator.py` file via the command line to specify the hyperparameters and start the retrieval-augmented generation process. Command-line example might be:
```
nohup python3 HopGenerator.py --model_name 'gpt-3.5-turbo-0125' --data_path 'quickstart_dataset/hotpot_example.jsonl' \
--retriever_name 'HopRetriever' --max_hop 4 --topk 20 --traversal bfs --mode common --label 'your_index_prefix_in_neo4j'  > nohup.txt  &
```
Please note that:
- Please refer to the `HopGenerator.py` to set the parameters.
- We provide `main_musique` and `main_hotpot` for MuSiQue and HotpotQA respectively.
- After running `main_musique` or `main_hotpot`, the  `result_dir` will add one more directory with:
    - `cache` dir filled with `.json` file, which logs the detailed answer and context contents for each question
    - one result file (with the retrieval context index and the generated response) in the format that can be directly evaluated by the specific evaluation tool in the corresponding benchmark.

## 6. Evaluation  
After completing Step 5, you will have the recalled contexts and generated responses for each question in the `.json`(hotpotqa)  or `.jsonl`(musique) file in `result_dir`, which is already ready for evaluation. This evaluation script depends on specific evaluation tools for the dataset, such as the Hotpot QA evaluation tool (you can refer to its repository for details).
