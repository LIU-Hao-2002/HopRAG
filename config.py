neo4j_notification_filter = ["DEPRECATION"]
exception_log_path = "exception_log.txt"
embed_model = "bge-base-zh-v1.5"
embed_model_dict = {"bge-base-zh-v1.5":"/path/to/bge"} 

embed_dim = 768

signal= "\n"     # the seperation for each doc in hotpot, customized to fit the data form
max_try_num = 2 # the attempt times for certain function calling
max_thread_num = 1  # Use 1 thread for API access; frequent requests or multiprocessing may cause errors.

LOG = True
DEBUG = False

# gpt-4o-mini
personal_base =  'https://api.openai.com/v1'
personal_key = "sk-...."  
default_gpt_model = "your model" 

neo4j_url = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "your password"
neo4j_dbname = "neo4j"

create_relation_query = """\
OPTIONAL MATCH (qa1) WHERE id(qa1) = $id1
OPTIONAL MATCH (qa2) WHERE id(qa2) = $id2
WITH qa1, qa2
WHERE qa1 IS NOT NULL AND qa2 IS NOT NULL
CREATE (qa1)-[:{}]->(qa2)
"""

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

### Sentence List
{sentences}

Your response must strictly follow the JSON format, avoiding unnecessary escapes, line breaks, or spaces. You should also pay extra attention to ensure that, except for the JSON and list formats themselves using double quotes ("), other instances of double quotes should be replaced with single quotes. For example, use '(How to Live) As Ghosts' instead of "(How to Live) As Ghosts".```json{{"Question List":["<Question 1>","<Question 2>",.....]}}```
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

### Sentences of News
{sentences}

Your response must strictly follow the JSON format, avoiding unnecessary escapes, line breaks, or spaces. You should also pay extra attention to ensure that, except for the JSON and list formats themselves using double quotes ("), other instances of double quotes should be replaced with single quotes. For example, use '(How to Live) As Ghosts' instead of "(How to Live) As Ghosts".```json{{"Question List":["<Question 1>","<Question 2>",”<Question 3>”,.....]}}```
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
```json{{“Title":"Inactive USAF Unit: 29th Military Airlift Squadron Disbanded in 1968"}}```

Document:
{doc_content}
Your response must strictly follow the JSON format, avoiding unnecessary escapes, line breaks, or spaces. You should also pay extra attention to ensure that, except for the JSON and list formats themselves using double quotes ("), other instances of double quotes should be replaced with single quotes. For example, use '(How to Live) As Ghosts' instead of "(How to Live) As Ghosts".```json{{“Title":"<title>"}}```

"""

create_entity_query = """
CREATE (node:{type} {{text: $text, keywords: $keywords, embed: $embed}}) RETURN id(node)
"""

create_pending2answerable='''
MATCH (a), (b)
WHERE id(a) = $id1 AND id(b) = $id2
CREATE (a)-[r:pen2ans_musique1000 {
    keywords: $keywords,
    embed: $embed,
    question: $answerable_question
}]->(b)
'''

create_abstract2answerable='''
MATCH (a), (b)
WHERE id(a) = $abstract_id AND id(b) = $id2
CREATE (a)-[r:pen2ans_musique1000 {
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
LIMIT 30
"""
retrieve_edge_dense_query="""
CALL db.index.vector.queryRelationships({index}, 30, {embedding}) YIELD relationship AS dense_edge, score AS dense_score
WITH dense_edge, dense_score
MATCH (startNode)-[dense_edge]->(endNode)
RETURN endNode, dense_edge, dense_score
ORDER BY dense_score DESC
LIMIT 30
"""

retrieve_node_sparse_query="""
CALL db.index.fulltext.queryNodes({index}, {keywords}) YIELD node AS sparse_node, score AS sparse_score
WITH sparse_node, sparse_score
RETURN sparse_node, sparse_score
ORDER BY sparse_score DESC
LIMIT 30
"""

retrieve_node_dense_query="""
CALL db.index.vector.queryNodes({index}, 30, {embedding}) YIELD node AS dense_node, score AS dense_score
WITH dense_node, dense_score
RETURN dense_node, dense_score
ORDER BY dense_score DESC
LIMIT 30
"""

expand_logic_query="""
MATCH (dense_node)-[r]-(logic_node)
where dense_node.text=$text
RETURN logic_node
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
Decision 1:[This information is not needed to answer the question]。In this case, you determine that you can answer the question even without the given background information, or the background information is not essential for answering the question. You should immediately return the decision in JSON format as follows:```json{{"Decision":"This information is not needed to answer the question"}}```
Note: If you determine this is Decision 1, return the result immediately without proceeding to Step 2. However, if you find that Step 2 is required, you must strictly follow the criteria for returning either Decision 2 or Decision 3.

Step 2: Choose a follow-up question:You have determined that answering the question strictly requires the given background information, but you realize that additional information is still needed. From the list of follow-up questions provided, select the one that is most helpful for gathering the remaining necessary information. Your decisions can include the following two types:
Decision 2:[Lacks appropriate follow-up questions]。In this case, none of the follow-up questions in the provided dictionary will help you answer the question. These follow-ups may seem related but cannot provide the critical information needed。 In this case, you should respond with:```json{{"Decision":"Lacks appropriate follow-up questions"}}```
Decision 3:[Choose a follow-up question]。Note: This situation is more strict; please make a careful judgment, do not make decisions hastily. In this situation, it is crucial that the answer to one follow-up question provides exactly the additional information needed to answer the question. Once you obtain the answer to this follow-up question, combined with the background information, you should be able to answer the question and finish the task. In this case, return your decision in JSON format as follows:```json{{"Decision":"Choose a follow-up question","Follow-up":"<follow-up content>"}}```

Example of Decision 1:
Question: Donnie Smith, who plays as a left-back for New England Revolution, belongs to what league featuring 22 teams?
Background Information: In Major League Soccer, several teams annually compete for secondary rivalry cups that are usually contested by only two teams, with the only exception being the Cascadia Cup, which is contested by three teams.
Follow-up Dictionary:{{"1":"Which leagues do relegated teams drop to?","2":"How are these cups or trophies comparable in the context of college football rivalries?","3":"What are most cups conceived as between teams?"}}
In this case, you determine that the background information does not help answer the question, as you can answer the question even without it. Your response should be:
```json{{"Decision":"This information is not needed to answer the question"}}```

Example of Decision 2:
Question:Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?
Background Information:Donald W. "Donnie" Smith (born December 7, 1990 in Detroit, Michigan) is an American soccer player who plays as a left back for New England Revolution in Major League Soccer.
Follow-up Dictionary:{{"1":"Which league does the New England Revolution belong to?","2":"Who is associated with the Midnight Ride mentioned in the sentence?"}}
In this case, you determine that the background information is indeed necessary for answering the question, so you proceed to Step 2. However, none of the provided follow-up questions will help you gather the necessary information. Therefore, your response should be:
```json{{"Decision":"Lacks appropriate follow-up questions"}}```

Example of Decision 3:
Question:Who was the gunman of the hostage crisis which Chris Reason was awarded the Graham Perkin Australian Journalist of the Year Award for his coverage of?
Background Information:He was awarded the Graham Perkin Australian Journalist of the Year Award for his coverage of the Lindt Cafe siege in December 2014.
Follow-up Dictionary:{{"1":"Who won the Graham Perkin Australian Journalist of the Year Award in 1993?","2":"Who was the gunman involved in the Sydney siege at the Lindt Cafe?","3":"In which city and country does Chris Reason work as a senior reporter and presenter?"}}
In this case, you determine that the background information is essential for answering the question, so you proceed to Step 2. After careful reasoning, you realize that follow-up 1 and follow-up 3 are related but do not provide the critical information. However, follow-up 2 will give you the needed details to answer the question. Therefore, you choose follow-up 2, and your response should be:
```json{{"Decision":"Choose a follow-up question","Follow-up":"Who was the gunman involved in the Sydney siege at the Lindt Cafe?"}}```


Now, please begin. Respond strictly in JSON format, avoiding unnecessary escapes, newlines, or spaces. You should also pay special attention: except for JSON and list formats, all instances of double quotes should be changed to single quotes, such as in 'How to Live as Ghosts'.
Question、Background Information、Follow-up Dictionary as follows:
Question:{query}
Background Information:{node_content}
Follow-up Dictionary:{choices}

"""

llm_choice_query_chunk = """
You are a question-answering robot. I will provide you with a main question that involves multiple pieces of information, as well as an additional auxiliary question. Your task is to answer the main question, but since the main question involves a lot of information that you may not know, you have the opportunity to use the auxiliary question to gather the information you need. However, the auxiliary question may not always be useful, so you need to assess the relationship between the auxiliary and the main question to determine whether or not to use it.

You need to assess whether the auxiliary question is completely irrelevant, indirectly relevant, or relevant and necessary for answering the main question. You can only return one of these three outcomes.

Please note that the main question will involve multiple background sentences, meaning that answering the main question requires the combination and reasoning of several pieces of information. However, you do not know which specific sentences are necessary to answer the main question. Your task is to assess whether the given auxiliary question is relevant and necessary, Indirectly relevant, or completely irrelevant in answering the main question.

Result 1: [Completely Irrelevant]. In this case, you determine that even without the information from the auxiliary question, you can still answer the main question, or the information in the auxiliary question is completely unrelated to the answer of the main question.
Result 2: [Indirectly relevant]. In this case, you find that the auxiliary question is related to the main question, but its answer is not part of the multiple pieces of information needed to answer the main question. The auxiliary question focuses on a related topic but does not provide critical information necessary for answering the main question.
Result 3: [Relevant and Necessary]. In this case, you find that the auxiliary question is a sub-question of the main question, meaning that without answering the auxiliary question, you will not be able to answer the main question. The information provided by the auxiliary question is necessary to answer the main question.

Example of Result 1:
Main Question: Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?
Auxiliary Question: What is the purpose of the State of the Union address presented by the President of the United States?
In this case, after careful consideration, you find that the auxiliary question does not help answer the main question. The auxiliary question is completely unrelated to the main question. Your response should be:
```json{{" Decision ":"Completely Irrelevant"}}```

Example of Result 2:
Main Question: Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?
Auxiliary Question: What is the significance of this league for second teams in the region?
In this case, you notice that both the main and auxiliary questions involve similar topics, but the auxiliary question focuses on the significance of the league, while the main question asks about the league to which a specific player belongs. Upon careful consideration, you find that the auxiliary question is related, but its answer does not provide any critical information to answer the main question. Your response should be: ```json{{" Decision ":" Indirectly relevant "}}```

Example of Result 3:
Main Question: Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?
Auxiliary Question: Which team does Donald W. 'Donnie' Smith play for in Major League Soccer?
In this case, after careful consideration, you find that the auxiliary question is indeed related to the main question. The auxiliary question is a sub-question of the main question and provides necessary information to answer the main question. Without the answer to the auxiliary question, you will not be able to answer the main question. Your response should be: ```json{{" Decision ":" Relevant and Necessary "}}```

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