neo4j_notification_filter = ["DEPRECATION"]
exception_log_path = "exception_log.txt"
embed_model = 'bge_en'
embed_model_dict = {"bge_en":"/path/to/your/bge_base_en"} 
local_model_name = "local_qwen_3_8b" # modelname from locally deployed vllm server
embed_dim = 768
llm_device='cuda:5'
reranker="/path/to/bge_reranker_base"
query_generator_model=local_model_name
traversal_model=local_model_name

signal= "\n\n"     # the seperation for each doc in hotpot, customized to fit the data form;
max_try_num = 2 # the attempt times for llm calling
max_thread_num = 1  # Use 1 thread for API access; frequent requests or multiprocessing may cause errors.

dataset_name="hotpot" # hotpot;musique;wiki
node_name=dataset_name+'_bgeen_qwen1b5' # hotpot_bgeen_qwen1b5
edge_name="pen2ans_"+node_name # pen2ans_hotpot_bgeen_qwen1b5

generator_label=node_name+'_' # hotpot_bgeen_qwen1b5_
node_dense_index_name=generator_label+'node_dense_index'
edge_dense_index_name=generator_label+'edge_dense_index'
node_sparse_index_name=generator_label+'node_sparse_index'
edge_sparse_index_name=generator_label+'edge_sparse_index'
LOG = True
DEBUG = False

local_base =  'http://localhost:8924/v1'
local_key = "EMPTY"

gpt_base = '' # don't add /chat/completions
gpt_key = ''
default_gpt_model = "gpt-4o-mini"

deployment_sign = {"gpt":{"base":gpt_base,"key":gpt_key,"default_model":default_gpt_model},
                   local_model_name:{"base":local_base,"key":local_key}}

neo4j_url = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "your_password"
neo4j_dbname = "neo4j"

# 'fixed' without summary ensures questions focus on the text itself; 'pending' without summary allows questions to explore other texts.

extract_template_fixed_eng="""
You are a journalist who is good at asking questions and proficient in both Chinese and English. Your task is to generate questions based on a few consecutive sentences from a news article or a biographical text. However, the answers to your questions should only come from these specific sentences, i.e., you should reverse-generate questions from a few sentences of the text. You will only have access to a few sentences, not the entire document. Focus on these consecutive sentences and ask relevant questions, ensuring that the answers come exclusively from these sentences.

Requirements:
1. Each question must include specific news elements (time, place, person) or other key characteristics to reduce ambiguity, clarify the context, and ensure self-containment.
2. You can try to omit or leave blanks in important parts of the sentence and form questions, but do not ask multiple questions about the same part of the sentence. You do not need to ask a question for every part of the sentence.
3. When asking about a part that has been omitted, the non-omitted information should be included in the question, as long as it does not affect the coherence of the question.
4. Different questions should focus on different aspects of the information in these sentences, ensuring diversity and representativeness.
5. All questions combined should cover all key points of the provided sentences, and the phrasing should be standardized.
6. Questions should be objective, fact-based, and detail-oriented. For example, ask about the time an event occurred, personal details of the subject, etc. Ensure that the answers to the questions come solely from these sentences.
7. If a part of the sentence has already been mentioned in a previous question, you should not ask about it again. That is, if the information from a sentence has already been covered in earlier questions, it should not be repeated. However, all information from the sentences must be covered by the questions, and if the sentences are long, the number of questions should increase to accommodate all information. There is no upper limit to the number of questions, but avoid repetition.

### Example of Sentence List
["Their eighth studio album, \"(How to Live) As Ghosts\", is scheduled for release on October 27, 2017."]
### Example of Answer
```json{{"Question List":["What's the name of their eighth album?","When was the album '(How to Live) As Ghosts' scheduled to be released?"]}}```

Your response must strictly follow the JSON format, avoiding unnecessary escapes, line breaks, or spaces. You should also pay extra attention to ensure that, except for the JSON and list formats themselves using double quotes ("), other instances of double quotes should be replaced with single quotes. For example, use '(How to Live) As Ghosts' instead of "(How to Live) As Ghosts".
### Example of Answer
```json{{"Question List":["What...?","Who....?",.....]}}```

The followings are your Sentences of News:
{sentences}
"""

extract_template_pending_eng="""
You are a journalist skilled in asking insightful questions and proficient in two languages. Your task is to generate follow-up questions based on a few consecutive sentences from a news article or biographical text. A follow-up question refers to a question whose answer is not found within the given sentences, but the answer may be inferred from the context before or after the given sentences, from related documents covering the same event, or from logical, causal, or temporal extensions of keywords within the given sentences. 

You will only have access to a few sentences, not the entire document. After reading the consecutive sentences, generate related questions ensuring that the answer is not contained within these specific sentences. You can try to predict what the reader might ask next after reading these sentences, but the answers to your questions should be as concise as possible, so it is better to focus on objective questions.

Requirements:
1. Each question must include specific news elements (time, place, person) or other key features to reduce ambiguity and ensure self-containment.
2. Different follow-up questions should focus on diverse, objective aspects of the overall event represented by these sentences, ensuring variety and representativeness. Prioritize objective questions.
3. Based on the given sentences, generate questions about details that involve causal relationships, parallelism, sequencing, progression, connections, and other logical aspects. Possible areas to explore include, but are not limited to: the background of the event, information, reasons, impacts, significance, development trends, or perspectives of the individuals involved.
4. Questions should be objective, factual, and detail-oriented. For example, inquire about the time an event occurred, or ask for personal information about the subject. However, ensure that the answers to your questions are *not* contained in these specific sentences.
5. Aim to generate as many questions as possible without repetition, but ensure that the answers to the questions do not appear in these sentences. There is no upper limit to the number of questions, but please avoid duplicating questions.

### Example of Sentence
"Their eighth studio album, \"(How to Live) As Ghosts\", is scheduled for release on October 27, 2017."
### Example of Answer
```json{{" Question List ":["Whose eighth studio album is '(How to Live) As Ghosts'?","How did the album '(How to Live) As Ghosts' perform?","How long did it take to make the album '(How to Live) As Ghosts'?"]}}```

Your response must strictly follow the JSON format, avoiding unnecessary escapes, line breaks, or spaces. You should also pay extra attention to ensure that, except for the JSON and list formats themselves using double quotes ("), other instances of double quotes should be replaced with single quotes. For example, use '(How to Live) As Ghosts' instead of "(How to Live) As Ghosts".

### Example of Answer
```json{{"Question List":["What...?","How...?","Who...?",.....]}}```

The followings are your Sentences of News:
{sentences}
"""

title_template_eng="""
You are a news editorial assistant skilled in titling documents, and you are proficient in two languages. Your task is to create a title in English for an English document. The title should be concise, clear, and accurately summarize the main theme of the news document. It should be engaging and make the reader want to read further.

Note that the title should provide a summary of the content of the news document. It must cover the key subject and details of the news, encapsulating the theme, but avoid being overly detailed or abstract. The title should reflect the characteristics of a typical news headline—brief, straightforward, and capable of sparking the reader’s interest.

### News Document and Title Example
Document: 
The 29th Military Airlift Squadron is an inactive United States Air Force unit.

 Its last was assigned to the 438th Military Airlift Wing, Military Airlift Command, stationed at McGuire Air Force Base, New Jersey.

 It was inactivated on 31 August 1968.
Title: 
```json{{"Title":"Inactive USAF Unit: 29th Military Airlift Squadron Disbanded in 1968"}}```

Your response must strictly follow the JSON format, avoiding unnecessary escapes, line breaks, or spaces. You should also pay extra attention to ensure that, except for the JSON and list formats themselves using double quotes ("), other instances of double quotes should be replaced with single quotes. For example, use '(How to Live) As Ghosts' instead of "(How to Live) As Ghosts".
your format:
```json{{"Title":"<title>"}}```

Document:
{doc_content}

"""

create_entity_query = """
CREATE (node:{type} {{text: $text, keywords: $keywords, embed: $embed}}) RETURN id(node)
"""

create_pending2answerable="""
MATCH (a), (b)
WHERE id(a) = $id1 AND id(b) = $id2
CREATE (a)-[r:"""+edge_name+""" {
    keywords: $keywords,
    embed: $embed,
    question: $answerable_question
}]->(b)
"""

create_abstract2answerable='''
MATCH (a), (b)
WHERE id(a) = $abstract_id AND id(b) = $id2
CREATE (a)-[r:'''+edge_name+''' {
    keywords: $keywords,
    embed: $embed,
    question: $answerable_question
}]->(b)
'''

create_node_dense_index_template = """
    CREATE VECTOR INDEX {name} IF NOT EXISTS
    FOR (m:{type})
    ON m.{property}
    OPTIONS {{indexConfig: {{
    `vector.dimensions`: {dim},
    `vector.similarity_function`: 'cosine'
    }}}}
"""
create_edge_dense_index_template = """
    CREATE VECTOR INDEX {name} IF NOT EXISTS
    FOR ()-[m:{type}]-()
    ON m.{property}
    OPTIONS {{indexConfig: {{
    `vector.dimensions`: {dim},
    `vector.similarity_function`: 'cosine'
    }}}}
"""

create_node_sparse_index_template='''
CREATE FULLTEXT INDEX {name} IF NOT EXISTS
FOR (m:{type})
ON EACH [m.{property}]
'''


create_edge_sparse_index_template='''
CREATE FULLTEXT INDEX {name} IF NOT EXISTS
FOR ()-[r:{type}]-()
ON EACH [r.{property}]
'''


retrieve_edge_sparse_query = """
CALL db.index.fulltext.queryRelationships({index}, {keywords}) YIELD relationship AS sparse_edge, score AS sparse_score
WITH sparse_edge, sparse_score
MATCH (startNode)-[sparse_edge]->(endNode)
RETURN endNode, sparse_edge, sparse_score
ORDER BY sparse_score DESC
LIMIT 40
"""
retrieve_edge_dense_query="""
CALL db.index.vector.queryRelationships({index}, 40, {embedding}) YIELD relationship AS dense_edge, score AS dense_score
WITH dense_edge, dense_score
MATCH (startNode)-[dense_edge]->(endNode)
RETURN endNode, dense_edge, dense_score
ORDER BY dense_score DESC
LIMIT 40
"""

retrieve_node_sparse_query="""
CALL db.index.fulltext.queryNodes({index}, {keywords}) YIELD node AS sparse_node, score AS sparse_score
WITH sparse_node, sparse_score
RETURN sparse_node, sparse_score
ORDER BY sparse_score DESC
LIMIT 40
"""

retrieve_node_dense_query="""
CALL db.index.vector.queryNodes({index}, 40, {embedding}) YIELD node AS dense_node, score AS dense_score
WITH dense_node, dense_score
RETURN dense_node, dense_score
ORDER BY dense_score DESC
LIMIT 40
"""

expand_logic_query="""
MATCH (dense_node)-[r]-(logic_node)
where dense_node.text=$text
RETURN logic_node
"""

expand_node_edge_query="""
MATCH (dense_node)-[out_edge]->(out_node)
where dense_node.text=$text
RETURN out_node, out_edge 
"""

get_out_edge_query="""
match (n)-[r]->(m)
where n.embed=$embed
and n.text=$text
return r as out_edge, m as out_node
"""

llm_choice_query = """
You are a question-answering bot. I will provide you with a question involving multiple pieces of information, a piece of background information, and a dictionary of follow-up questions derived from this background information. You need to decide the next step based on the required question and background information to ensure you gather all the information necessary to answer the question, without any omissions. Due to limited information, you may need to ask follow-up questions based on the background information in order to further clarify any details needed to answer the question. Therefore, I allow you to choose follow-up questions when more information is required to answer the question, but you can only select the one follow-up question from the list provided that is most helpful for answering the question.
Your decision-making process should follow two steps: The first step is to determine whether answering the question strictly requires the given background information. If not, return the result immediately. If it does, proceed to the second step, where you decide whether to ask further follow-up questions, allowing for two possible outcomes. Below is a detailed description of both steps. You can only return one of the three decisions!

Step 1: Determine if the background information is strictly required to answer this question.
Decision 1:[Not Needed].In this case, you determine that you can answer the question even without the given background information, or the background information is not essential for answering the question. You should immediately return the decision in JSON format as follows:```json{{"Decision":"Not Needed"}}```
Note: If you determine this is Decision 1, return the result immediately without proceeding to Step 2. However, if you find that Step 2 is required, you must strictly follow the criteria for returning either Decision 2 or Decision 3.

Step 2: Follow-up:You have determined that answering the question strictly requires the given background information, but you realize that additional information is still needed. From the list of follow-up questions provided, select the one that is most helpful for gathering the remaining necessary information. Your decisions can include the following two types:
Decision 2:[Lack Queries].In this case, none of the follow-up questions in the provided dictionary will help you answer the question. These follow-ups may seem related but cannot provide the critical information needed. In this case, you should respond with:```json{{"Decision":"Lack Queries"}}```
Decision 3:[Follow-up].Note: This situation is more strict; please make a careful judgment, do not make decisions hastily. In this situation, it is crucial that the answer to one follow-up question provides exactly the additional information needed to answer the question. Once you obtain the answer to this follow-up question, combined with the background information, you should be able to answer the question and finish the task. In this case, return your decision in JSON format as follows:```json{{"Decision":"<the index for the selected follow-up question>"}}```

Example of Decision 1:
```json
{{"Decision":"Not Needed"}}
```

Example of Decision 2:
```json
{{"Decision":"Lack Queries"}}
```

Example of Decision 3:
```json
{{"Decision":"2"}}
```


Now, please begin. Respond strictly in JSON format, avoiding unnecessary escapes, newlines, or spaces. You should also pay special attention: except for JSON and list formats, all instances of double quotes should be changed to single quotes, such as in 'How to Live as Ghosts'.
Question、Background Information、Follow-up Dictionary as follows:
Question:{query}
Background Information:{node_content}
Follow-up Dictionary:{choices}

"""



llm_choice_query_chunk2="""  
You are a question answering robot and I will give you a question with multiple information points and a sentence of background information. Depending on the question you need to answer, you need to determine whether this background information is Completely Irrelevant to answering the question, Indirectly Relevant, or Relevant and Necessary. You can only return one of these three results.
Please note that the question I give you must involve multiple sentences of background information, that is, the answer to the question must require the coordination and reasoning between multiple sentences to get the answer. But you don't know exactly what information sentence is needed to answer the question, you just need to decide whether the sentence given to you is Relevant and Necessary to answer the multi-information question, Indirectly Relevant, or Completely Irrelevant.
Result 1: [Completely Irrelevant]. You find that you can answer the question without knowing the background information, or that the background information you are given has nothing to do with the answer to the question.
Result 2: [Indirectly Relevant]. At this point, you find that the background information given to you has a certain relationship with the answer to the question, but you can't rigorously dig out the information from the background information that will really help answer the question, for example, maybe the sentence focuses on other aspects of a similar topic. The background information given to you is not necessary to answer the question.
Result 3: [Relevant and Necessary]. At this point, you find that although you cannot answer the question without this information point, which means that the background information given to you is indeed relevant to the question and necessary to answer the question, and you cannot answer the question without this information.
Example of result 1:
Question: Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?
Background information: He was awarded the Graham Perkin Australian Journalist of the Year Award for his coverage of the Lindt Cafe siege in  December 2014.
In this case, after careful consideration, you find that the background information given to you does not help you answer the question, that is, you can answer the question even if you do not know the background information. This background information has nothing to do with the problem. Your response should be:
```json{{"Decision":"Completely Irrelevant"}}```

Example of result 2:
Question: Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?
Background information: In Major League Soccer,  several teams annually compete for secondary rivalry cups that are usually contested by only two teams,  with the only exception being the Cascadia Cup, which is contested by three teams.
In this case, you first find that the background information involves similar information to the question, but the background information focuses on the league format and the question focuses on the league a player belongs to. After careful consideration, you find that the background information given to you has something to do with the answer to the question, but you can't rigorously dig out the background information that will actually help you answer the question. Your response should be:
```json{{"Decision":"Indirectly Relevant"}}```

Example of result three:
Question: Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?
Background information: Donald W. "Donnie" Smith (born December 7, 1990 in Detroit,  Michigan) is an American soccer player who plays as a left back for New England Revolution in Major League Soccer.
In this case, after careful consideration, you find that the background information given to you is indeed relevant to the question and necessary to answer the question, and that you would not be able to answer the question without this background information. Your response should be:
```json{{"Decision":"Relevant and Necessary"}}```

Start by replying strictly in json format, avoiding unnecessary escapes, line breaks, and white space. You need to note that json and the list format requires English double quotes "
Question and background information are as follows:
Question: {query}
Background information: {node_content}
"""
llm_choice_query_chunk="""
你是一个问答机器人，我会给你一个涉及多个信息点的问题和一句背景信息。你需要根据需要回答的问题，来判断这个背景信息是否对于回答这个问题是Completely Irrelevant，还是Indirectly Relevant，还是Relevant and Necessary。你只能返回这三类结果中的一个。
请你注意，我给你的问题一定涉及到多句背景信息，即想要回答这个问题必须需要多个句子之间的互相配合和推理才能得到答案。但是你并不知道这个回答这个问题究竟需要哪些信息句子，你只需要判断当下给你的这句话是否对于回答该多信息点问题是否相关且必要，或相关但不必要，或是完全无关。
结果一：【Completely Irrelevant】。此时你发现即使不知道给你的背景信息，也能回答这个问题，或是给你的背景信息和这个问题的答案没有一点关系。
结果二：【Indirectly Relevant】。此时你发现给你的背景信息和这个问题的答案有一定关系，但是你无法严谨地从这个背景信息中挖掘出真正有助于回答这个问题的信息，例如有可能这句话关注的是类似话题的别的侧重点。即给你的背景信息不是回答这个问题所必要的。
结果三：【Relevant and Necessary】。此时你发现，虽然在只获得这个信息点的情况下无法回答这个问题，但是如果没有这个背景信息，你必定将无法回答这个问题。也就是给你的背景信息是回答这个问题所必要的。

结果一的例子：
问题：Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?
背景信息：He was awarded the Graham Perkin Australian Journalist of the Year Award for his coverage of the Lindt Cafe siege in December 2014.
在这个情况下，你通过仔细的考量，发现给你的背景信息并不能帮助你回答这个问题，也就是即使你不知道这个背景信息也能回答这个问题。这个背景信息和问题没有什么关系。此时你的回复应该是：
```json{{"Decision":"Completely Irrelevant"}}```

结果二的例子：
问题：Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?
背景信息：In Major League Soccer, several teams annually compete for secondary rivalry cups that are usually contested by only two teams, with the only exception being the Cascadia Cup, which is contested by three teams.
在这个情况下，你先发现背景信息与该问题都涉及到了类似的信息，但是这个背景信息关注联赛的赛制，问题关注一名球员的所属联盟。你通过仔细的考量，发现给你的背景信息和这个问题的答案有一定关系，但是你无法严谨地从这个背景信息中挖掘出真正有助于回答这个问题的信息。此时你的回复应该是：
```json{{"Decision":"Indirectly Relevant"}}```

结果三的例子：
问题：Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?
背景信息：Donald W. "Donnie" Smith (born December 7, 1990 in Detroit, Michigan) is an American soccer player who plays as a left back for New England Revolution in Major League Soccer.
在这个情况下，你通过仔细的考量，发现给你的背景信息确实是与问题相关，且是回答这个问题所必要的，并且如果没有这个背景信息，你将无法回答这个问题。此时你的回复应该是：
```json{{"Decision":"Relevant and Necessary"}}```

下面请你开始，严格按照json格式回复，避免不必要的转义、换行和空白。你需要额外注意，除了json和列表格式本身需要英文双引号"外，其余要使用双引号的情况都改成英文单引号。例如文本中用'(How to Live) As Ghosts'
问题、背景信息如下：
问题：{query}
背景信息：{node_content}
"""

llm_node_choice_prompt="""
You are a question-answering bot. You need to answer a multi-hop query that touches a lot of information. I will give you only one text chunk. Of course you cannot answer the multi-hop query with the only text chunk alone, for lack of other useful information. Even so, the text chunk might be completely irrelevant to the multi-hop query.
However, you need to judge whether this text chunk is helpful for you to answer the multi-hop query. If it is helpful, you can save the information and wait for the follow-up information, since this text chunk is absolutely necessary to answer the multi-hop query. If it is helpless, you need to discard this redundant text chunk and wait for other information that is actually useful to answer the multi-hop query.
So your task is to judge whether the text chunk is helpful to answer the multi-hop query. 
Given the user multi-hop query, you can make 2 decisions on the text chunk: [helpful] or [helpless]. You can only return one of these two decisions!

Example of Decision 1:
```json
{{"Decision":"helpful"}}
```

Example of Decision 2:
```json
{{"Decision":"helpless"}}
```

Respond strictly in JSON format, avoiding unnecessary escapes, newlines, or spaces. 
User multi-hop query, text chunk are listed as follows:
multi-hop query:{query}
text chunk:{node_content}
Now, please begin. 
"""

llm_edge_choice_prompt="""
You are a question-answering bot. You need to answer a multi-hop query that touches a lot of information. Without any background, you cannot answer the multi-hop query for lack of any useful information. However, I can give you one follow-up question, which is the only chance you can use to answer the multi-hop query. 
Be careful! The follow-up question might not be necessarily helpful. It might be completely irrelevant to the query, or similar with it but actually unhelpful to the query([helpless]). Or it might lead to one of the many necessary and helpful pieces of information you need to answer the multi-hop query.([helpful])
You need to judge whether this follow-up question is helpful for you to answer the multi-hop query. If you think it is helpful, I will answer the follow-up question and it can help you answer the multi-hop query (even though the answer to the follow-up question might not be sufficient on its own, but every helpful bit counts). If you think it is helpless, you need to discard this redundant follow-up question and you will be rewarded, since you wisely save your only chance and wait for other information that is actually useful to answer the multi-hop query. 
Given the user multi-hop query, you can make 2 decisions on the follow-up question: [helpful] or [helpless]. You can only return one of these two decisions!

Example of Decision 1:
```json
{{"Decision":"helpful"}}
```

Example of Decision 2:
```json
{{"Decision":"helpless"}}
```

Respond strictly in JSON format, avoiding unnecessary escapes, newlines, or spaces. 
User multi-hop query and follow-up question are listed as follows:
multi-hop query:{query}
follow-up question:{question}
Now, please begin. 
"""

llm_choice_query_edge = """
You are a question-answering robot. I will provide you with a main question that involves multiple pieces of information, as well as an additional auxiliary question. Your task is to answer the main question, but since the main question involves a lot of information that you may not know, you have the opportunity to use the auxiliary question to gather the information you need. However, the auxiliary question may not always be useful, so you need to assess the relationship between the auxiliary and the main question to determine whether or not to use it.

You need to assess whether the auxiliary question is Completely Irrelevant, Indirectly Relevant, or Relevant and Necessary for answering the main question. You can only return one of these three outcomes.

Please note that the main question will involve multiple background sentences, meaning that answering the main question requires the combination and reasoning of several pieces of information. However, you do not know which specific sentences are necessary to answer the main question. Your task is to assess whether the given auxiliary question is relevant and necessary, Indirectly relevant, or completely irrelevant in answering the main question.

Result 1: [Completely Irrelevant]. In this case, you determine that even without the information from the auxiliary question, you can still answer the main question, or the information in the auxiliary question is completely unrelated to the answer of the main question.
Result 2: [Indirectly Relevant]. In this case, you find that the auxiliary question is related to the main question, but its answer is not part of the multiple pieces of information needed to answer the main question. The auxiliary question focuses on a related topic but does not provide critical information necessary for answering the main question.
Result 3: [Relevant and Necessary]. In this case, you find that the auxiliary question is a sub-question of the main question, meaning that without answering the auxiliary question, you will not be able to answer the main question. The information provided by the auxiliary question is necessary to answer the main question.

Example of Result 1:
Main Question: Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?
Auxiliary Question: What is the purpose of the State of the Union address presented by the President of the United States?
In this case, after careful consideration, you find that the auxiliary question does not help answer the main question. The auxiliary question is completely unrelated to the main question. Your response should be:
```json{{"Decision":"Completely Irrelevant"}}```

Example of Result 2:
Main Question: Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?
Auxiliary Question: What is the significance of this league for second teams in the region?
In this case, you notice that both the main and auxiliary questions involve similar topics, but the auxiliary question focuses on the significance of the league, while the main question asks about the league to which a specific player belongs. Upon careful consideration, you find that the auxiliary question is related, but its answer does not provide any critical information to answer the main question. Your response should be: 
```json{{"Decision":"Indirectly Relevant"}}```

Example of Result 3:
Main Question: Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?
Auxiliary Question: Which team does Donald W. 'Donnie' Smith play for in Major League Soccer?
In this case, after careful consideration, you find that the auxiliary question is indeed related to the main question. The auxiliary question is a sub-question of the main question and provides necessary information to answer the main question. Without the answer to the auxiliary question, you will not be able to answer the main question. Your response should be: 
```json{{"Decision":"Relevant and Necessary"}}```

Now please strictly follow the JSON format in your response, avoiding unnecessary escapes, line breaks, or spaces. Additionally, please note that, except for the JSON and list formats, you should replace all double quotes with single quotes. For example, use '(How to Live) As Ghosts'
Main Question, Auxiliary Question as follows:
Main Question: {query}
Auxiliary Question: {question}

"""

shortest_path_query ="""
MATCH (n) 
WHERE n.text=$text1
WITH n
MATCH (m)
WHERE m.text=$text2 AND id(n) <> id(m)
WITH n, m
MATCH p = shortestPath((n)-[*]-(m))
RETURN length(p) as length
"""

query_reformulation_template='''
You are a query reformulation robot. I will provide you with a multi-hop query that touches multiple information. Your task is to break down the query into multiple sub-queries, each of which should be a single-hop query. The sub-queries should be related to each other and can be answered in sequence. You need to ensure that the sub-queries are clear and concise, and that they can be answered independently. Return as few subqueries as possible, but make sure that all the information in the original query is covered.
Your response must strictly follow the JSON format, avoiding unnecessary escapes, line breaks, or spaces. You should also pay extra attention to ensure that, except for the JSON and list formats themselves using double quotes ("), other instances of double quotes should be replaced with single quotes. For example, use '(How to Live) As Ghosts' instead of "(How to Live) As Ghosts".

### Example of Answer
```json{{"Subqueries":["What...?","How...?",.....]}}```

The followings are your multi-hop query:
{query}
'''