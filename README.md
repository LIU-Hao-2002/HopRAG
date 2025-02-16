# Introduction  
HopRAG is a Graph-based RAG project that stores text in a Neo4j graph database, builds an index, and performs retrieval. It can be used for specific RAG evaluation datasets. 
Note:All paths in the project files need to be replaced with your local paths.

## 1. Prepare the Dataset  
Obtain the test set file and preprocess it using the `process_data` function in the `data_preprocess.py` file. The preprocessing will write the questions from the test set as `.txt` files into the specified directory.

## 2. Build Nodes  
Run the `main_nodes` function in the `HopBuilder.py` file.

## 3. Build Edges  
Run the `main_edges` function in the `HopBuilder.py` file.

## 4. Build the Index  
Create a `QABuilder` class in `HopBuilder.py` and run its `create_index` method.

## 5. Retrieval and Generation  
Once the database is built, run the `main` function in the `HopGenerator.py` file via the command line to start the retrieval and generation process. Command-line examples can be found at the end of `HopGenerator.py`.

## 6. Evaluation  
After completing Step 5, you will have the recalled contexts and generated responses for each question. You can perform evaluation at this stage. This evaluation script depends on specific evaluation tools for the dataset, such as the Hotpot QA evaluation tool (you can refer to its repository for details).
