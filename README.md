# Introduction  
HopRAG is a Graph-based RAG project that stores passages in a Neo4j graph database, builds an index, and performs retrieval. It can be used for specific RAG evaluation datasets. 
Note:All paths in the project files need to be replaced with your local paths.

## 1. Prepare the Dataset  
Obtain the test set file and preprocess it using the `process_data` function in the `data_preprocess.py` file. The preprocessing will write the passages for all the questions from the test set as `.txt` files into the specified directory. You might need to revise it according to the format of the specific dataset. 

## 2. Build Nodes  
Run the `main_nodes` function in the `HopBuilder.py` file.

## 3. Build Edges  
Run the `main_edges` function in the `HopBuilder.py` file.

## 4. Build the Index  
Create a `QABuilder` class in `HopBuilder.py` and run its `create_index` method.

## 5. Retrieval and Generation  
Once the database is built, run the `main` function in the `HopGenerator.py` file via the command line to start the retrieval and generation process. Command-line example might be:
```
nohup python3 HopGenerator.py --model_name 'gpt-3.5-turbo-0125' --data_path '/path/to/your/questions.jsonl' \
--retriever_name 'HopRetriever' --max_hop 4 --topk 20 --traversal bfs --mode common --label 'your_index_prefix_in_neo4j'  > nohup.txt  &
```

## 6. Evaluation  
After completing Step 5, you will have the recalled contexts and generated responses for each question. You can perform evaluation at this stage. This evaluation script depends on specific evaluation tools for the dataset, such as the Hotpot QA evaluation tool (you can refer to its repository for details).
