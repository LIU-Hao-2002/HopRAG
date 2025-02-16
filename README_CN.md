# 介绍
HopRAG是一个Graph RAG项目，基于neo4j图数据库存储文本，建立索引并进行检索。可以用于特定RAG评测集。项目文件中的所有路径需要进行本地替换。

## 1. 准备数据集
获取测试集文件，通过data_preprocess.py内的process_data函数对文件进行采样与预处理。预处理会将测试集内的问题以txt的文件的形式写入指定目录。

## 2. 建立节点
运行HopBuilder.py文件内的main_nodes函数

## 3. 建立边
运行HopBuilder.py文件内的main_edges函数

## 4. 建立索引
创建一个HopBuilder.py中的QABuilder类，运行其create_index方法

## 5. 检索生成
在数据库构建好之后，通过命令行运行HopGenerator.py的main函数，开始进行检索与生成。命令行示例可见于HopGenerator.py末尾。

## 6. 评测
在结束第五步后得到了每道题目的召回context和生成的response，可以进行评测。该部分评测脚本依赖于具体评测集的评测工具，例如hotpot qa可参见其代码仓