import sys
import os
from openai import OpenAI
import json
import pickle
from config import *
import torch
import random
from tqdm import tqdm
import re
import time
from paddlenlp import Taskflow
from sentence_transformers import SentenceTransformer
import numpy as np
import jsonlines
from typing import List, Tuple, Dict, Set

def sparse_similarity(a:Set, b:Set):
    return len(a.intersection(b))/len(a.union(b))

def try_run(func, *args, **kwargs):
    retry = 0
    while retry < max_try_num:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retry += 1
            with open(exception_log_path, "a") as file:
                file.write(f"Exception: {e}\n")
                file.write(f"Commandline: {sys.argv}\n")
            time.sleep(3)
    else:
        with open(exception_log_path, "a") as file:
            file.write(f"--FAIL--\n")
            file.write(f"Commandline: {sys.argv}\n")
        #exit(555)
        return None,None,None

def replace_newlines(match):
    # Replace \n and \r in the matched string
    return match.group(0).replace('\n', '\\n').replace('\r', '\\r')

def clean_json_str(json_str: str) -> str:
    """
    The generated JSON format may be non-standard, perform replacement processing first.
    :param json_str:
    :return:
    """
    # Remove code block markers ```
    # Replace None with null in the JSON string

    json_str = json_str.replace("None","null")
    if not json_str.startswith('```') and '```' in json_str:
        json_str = '```'+json_str.split('```')[1]
    if json_str.startswith("```") and not json_str.endswith("```"):
        json_str += "```"
    match = re.search(r'```json(.*?)```', json_str, re.DOTALL)
    if match:
        json_str = match.group(1)
    match = re.search(r'```(.*?)```', json_str, re.DOTALL)
    if match:
        json_str = match.group(1)
    # Replace \n and \r in the matched string
    json_str = re.sub( r'("(?:\\.|[^"\\])*")', replace_newlines, json_str)
    # Remove trailing commas after key-value pairs
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    # Restore the missing commas
    json_str = re.sub(r'\"\s+\"', '\",\"', json_str)
    # Inplacement of True and False
    json_str = json_str.replace("True","true")
    json_str = json_str.replace("False","false")
    return json_str

def txt2obj(text):  
    try: 
        text = clean_json_str(text) 
        text = text.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'")
        text = text.replace('`','')
        return json.loads(text) 
    except Exception as e:
        if LOG:
            print(e)
        return None




def get_title_keywords_eng(title_template, doc)->Tuple[str,Set[str]]:
    chat = []
    chat.append({"role": "user", "content": title_template.format(doc_content=doc)})
    title, chat = get_chat_completion(chat, keys=["Title"])
    if len(title)==0:
        title=doc[:20]
    keywords=get_ner_eng(title)
    if len(keywords)==0:
        keywords=title.replace(',',"").replace('，',"").replace('。','').replace('.','')
        keywords=set(keywords)
        return title,keywords
    return title,set(keywords)

def get_question_list(extract_template, sentences)->List[str]:
    chat = []
    chat.append({"role": "user", "content": extract_template.format(sentences=sentences)})
    question_list, chat = get_chat_completion(chat, keys=["Question List"])
    return question_list


def get_ner_eng(text):
    ner_task = Taskflow("pos_tagging")
    results = ner_task(text)
    filtered = []
    for result in results:
        entity, mode = result
        if mode not in [
            "w",  # Punctuation marks
            "c",  # Conjunctions
            "f",  # Directional words
            "ad", # Adverbs
            "q",  # Quantifiers
            "u",  # Particles
            "s",  # Locative words
            "vd", # Verbal adverbs
            "an", # Noun-adjective compound
            "r",  # Pronouns
            "xc", # Other function words
            "vn", # Noun-verb compounds
            "d",  # Adverbs
            "p",  # Prepositions
        ]:
            filtered.append(entity)
    filtered = list(set(filtered))
    return filtered

def load_embed_model(model_name):
    if model_name in embed_model_dict:
        return SentenceTransformer(embed_model_dict[model_name],device='cuda:0')  #
    else:
        raise NotImplementedError

def get_doc_embeds(documents, model):
    with torch.no_grad():
        embeddings = model.encode(documents, normalize_embeddings=True, device='cuda:0').tolist() # 
    return embeddings

def _get_chat_completion(chat, return_json=True, model=default_gpt_model, max_tokens=2048, keys=None):
    if not isinstance(chat, list):
        chat = [{"role": "user", "content": chat}]
    client = OpenAI(api_key=personal_key, base_url=personal_base)
    chat_completion = client.chat.completions.create(model=model,
                                                   messages=chat,
                                                   response_format={"type": "json_object" if return_json else "text"},
                                                   max_tokens=max_tokens,
                                                   temperature=0.1,
                                                   frequency_penalty=0.0,
                                                   presence_penalty=0.0)
    if LOG:
        print(chat_completion.choices[0].message.content)
    chat = chat + [{"role": "assistant", "content": chat_completion.choices[0].message.content}]
    if not return_json:
        return chat_completion.choices[0].message.content, chat
    obj = txt2obj(chat_completion.choices[0].message.content)
    obj = tuple([obj[key] for key in keys if key in obj]) #
    return *obj, chat

def get_chat_completion(chat, return_json=True, model=default_gpt_model, max_tokens=2048, keys=None):
    return try_run(_get_chat_completion, chat, return_json, model, max_tokens, keys)

def pending_dot_answerable(pending_df,answerable_df):
    pending=np.array(pending_df['embedding'].tolist())
    answerable=np.array(answerable_df['embedding'].tolist())
    if torch.cuda.is_available():
        pending=torch.tensor(pending).cuda()
        answerable=torch.tensor(answerable).cuda()
        dense_similarity=pending.mm(answerable.T).cpu().numpy()
    else:
        dense_similarity=pending.dot(answerable.T)
    outcome=dense_similarity.flatten().tolist()
    del pending,answerable,dense_similarity
    torch.cuda.empty_cache()
    return outcome

def sparse_similarities_df(df)->Dict[Tuple[str,str],float]:
    if os.path.exists('/path/to/cache/sparse_similarities_result.pkl'):
        with open('/path/to/cache/sparse_similarities_result.pkl','rb') as file:
            return pickle.load(file)
    docs_keywords=df['keywords'].astype(str).unique()
    sparse_similarities={}
    for i in range(len(docs_keywords)):
        for j in range(i,len(docs_keywords)):  
            sparse_similarities[(docs_keywords[i],docs_keywords[j])]=sparse_similarity(set(eval(docs_keywords[i])),set(eval(docs_keywords[j])))
            sparse_similarities[(docs_keywords[j],docs_keywords[i])]=sparse_similarity(set(eval(docs_keywords[i])),set(eval(docs_keywords[j])))
    return sparse_similarities


def process_data(source_path, docs_dir, output_path, nums):
    doc2id = {}
    
    # Open and load the source data
    with open(source_path, 'r') as f:
        data = json.load(f)
    
    # Randomly sample the desired number of entries
    chosen = random.sample(data, nums)
    
    # Process the chosen entries and create text files for documents
    for temp in tqdm(chosen):
        _id = temp['_id']
        context = temp['context']
        for title, sentences in context:
            doc = "\n\n".join(sentences)
            if doc not in doc2id:
                doc2id[doc] = title

    # Ensure the docs_dir exists
    os.makedirs(docs_dir, exist_ok=True)
    
    # Write each document to a text file
    for doc, _id in doc2id.items():
        if '/' in _id:
            _id = _id.replace('/', '_')
        with open(os.path.join(docs_dir, f'{_id}.txt'), 'w') as f:
            f.write(doc)
    
    # Print completion message
    print(f'done: all text files saved to directory {docs_dir}')
    
    # Write the selected data to a jsonlines file
    with jsonlines.open(output_path, mode='w') as writer:
        for result in chosen:
            writer.write(result)


if __name__ == "__main__":
    print(get_chat_completion([{"role": "user", "content": "What is the capital of China?"}]))