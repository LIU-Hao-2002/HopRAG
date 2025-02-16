from tool import *
from HopRetriever import HopRetriever
from typing import List
import argparse
import json
import os
from tqdm import tqdm
from loguru import logger
import time
parser = argparse.ArgumentParser()
# Model related options
parser.add_argument('--model_name', default='gpt-3.5-turbo', help="Name of the model to generate responses,not used in retriever")
# Dataset related options
parser.add_argument('--data_path', default='/path/to/hotpot_dev_distractor_v1_1000.jsonl', help="Path to the dataset")
parser.add_argument('--retriever_name', default="base", help="Name of the retriever")

# arguments for HopRAG  
parser.add_argument('--max_hop',default=5,type=int)
parser.add_argument('--start_layer',default=0,type=int)
parser.add_argument('--max_layer',default=2,type=int)
parser.add_argument('--entry_type',default='node')
parser.add_argument('--trim', action='store_true', default=False)
parser.add_argument('--hybrid', action='store_true', default=False)
parser.add_argument('--tol',default=20,type=int)
parser.add_argument('--mock_dense',action='store_true',default=False)
parser.add_argument('--mock_sparse',action='store_true',default=False)
parser.add_argument('--mode',default='common',type=str,help='common,sp,cache')
parser.add_argument('--label',type=str,default='musique1000_',help='the label of the dataset and the index')
parser.add_argument('--topk',default=8,type=int)
parser.add_argument('--traversal',default='bfs',type=str,help='bfs,bfs_sim_node,dfs')

generate_prompt="""You are a helpful assistant. Please answer my question given the following context. If the context lacks necessary information to answer the question, please try your best to reason and answer in the right format. You have to give an answer no matter what.

Please reply in a json format with only your answer. Do not repeat the context. The correct format is as follows:
```json{{"answer": "<your answer>"}}```

Example1:
Question: What is the name of the county that Cari Roccaro is from?, 
Context: ["East Islip is a hamlet and CDP in the Town of Islip, Suffolk County, New York, United States.","Cari Elizabeth Roccaro (born July 18, 1994) is an American soccer defender from East Islip, New York."]
Your answer should be in the format:
```json{{"answer": "Suffolk"}}```

Now please start.Answer this question in as fewer number of words as possible!!
Question: {query}
Context:{context} """

class RagPipeline:
    def __init__(self, args):
        self.args = args
        self.retriever = self._get_retriever()
        self.questions=None



    def _get_retriever(self):
        if self.args.retriever_name == "HopRetriever":
            retriever = HopRetriever(max_hop=self.args.max_hop,entry_type=self.args.entry_type,if_trim=self.args.trim,
                                     if_hybrid=self.args.hybrid,tol=self.args.tol,mock_dense=self.args.mock_dense,mock_sparse=self.args.mock_sparse,
                                     label=self.args.label,topk=self.args.topk,traversal=self.args.traversal)
        else:
            raise ValueError(f"Unknown retriever: {self.args.retriever_name}")
        return retriever

    def retrieve(self,query)->List[str]:
        return self.retriever.search_docs(query)
    
    
    def retrieve2(self, query)->List[str]:
        #musique: convert the SP of each question to context
        question_dict=[x for x in self.questions if x['question']==query][0]
        paragraphs=question_dict['paragraphs']
        context=[x['paragraph_text'] for x in paragraphs if x['is_supporting']==True]
        return context

    
    def retrieve3(self,query:str)->List[str]:
        '''Get the topk portion of the existing context'''
        topk=self.args.topk
        question_dict=[x for x in self.questions if x['question']==query][0]
        id=question_dict['id']
        with open(f'/path/to/musique/cache/{id}.json','r') as f:
            old_context=json.load(f)['context']
        old_context=old_context[:topk]
        return old_context
    
    def rag(self,query:str):
        if self.args.mode=='sp':
            context=self.retrieve2(query) # list
        elif self.args.mode=='cache':
            context=self.retrieve3(query)
        else:
            context=self.retrieve(query)
        chat=[]
        chat.append({"role": "user", "content": generate_prompt.format(query=query,context=context)})
        answer, chat = get_chat_completion(chat, keys=["answer"],model=self.args.model_name)
        return answer,context # list
    
def get_sentenceid2idx_musique(question_path):
    dir_="/path/to/musique/data/musique_ans_v1.0_dev_1000_sentence2titid.json"
    if os.path.exists(dir_):
        with open(dir_,'r') as f:
            return json.load(f)
    else:
        sentenceid2idx={}
        with open(question_path,'r') as f:
            for line in f:
                dp=json.loads(line)
                context=dp['paragraphs']
                id=dp['id']
                for dic in context:
                    sentenceid2idx[id+'__'+dic['paragraph_text']]=dic['idx']
        with open(dir_,'w') as f:
            json.dump(sentenceid2idx,f)
        return sentenceid2idx

def main():
    args = parser.parse_args()
    rag_pipeline = RagPipeline(args)
    questions_path=args.data_path
    questions=[]
    with open(questions_path,'r') as f:
        for line in f:
            questions.append(json.loads(line))
    rag_pipeline.questions=questions
    result_dir=f"/path/to/musique/musique/output/{args.retriever_name}_{args.model_name}_{args.max_hop}_mock_dense_{args.mock_dense}_mock_sparse_{args.mock_sparse}_mode_{args.mode}_topk_{args.topk}_traversal_{args.traversal}"
    id2json={}
    if os.path.exists(result_dir):
        cache_dir=f"{result_dir}/cache"
        for file in os.listdir(cache_dir):
            with open(f"{cache_dir}/{file}",'r') as f:
                id2json[file.replace('.json','')]=json.load(f)
        result_dir=result_dir+'_1'
        print(f'!! load {len(id2json)} cache !!')
    os.makedirs(result_dir,exist_ok=True)
    result_cache_dir=f"{result_dir}/cache"
    os.makedirs(result_cache_dir,exist_ok=True)
    result=[]# to dump jsonl
    sentenceid2idx=get_sentenceid2idx_musique(questions_path)
    contexts=[]
    for data in tqdm(questions,desc='processing questions musique'):
        _id=data['id']
        query=data['question']
        if _id in id2json:
            response=id2json[_id]['response']
            context=id2json[_id]['context']
        else:
            try:
                response,context=rag_pipeline.rag(query)
                if context is None:
                    logger.info(f"{_id} context is None")
                    context=[]
                contexts.append(context)
            except Exception as e:
                logger.info(f"{_id} error:{e}")
                response='I don\'t know because of some errors'
                context=[]
                time.sleep(3)
        with open(f"{result_cache_dir}/{_id.replace('/','_')}.json",'w') as f:
            json.dump({'response':response,'context':context},f)
        # Since the scores in musique are based on index matching within the question, but the recalled sentences may come from other questions, update the index of these sentences to be above 100
        idx=[]
        count=0
        for sentence in context:
            if _id+'__'+sentence in sentenceid2idx:
                idx.append(sentenceid2idx[_id+'__'+sentence])
            else:
                idx.append(100+count)
                count+=1
        logger.info(f"question {_id} has {count} sentences not in the original question")
        result.append({'id':_id,'predicted_answer':response,'predicted_support_idxs':idx,'predicted_answerable':True})
    avg_context_length=sum([len(''.join(context)) for context in contexts])/len(contexts)
    with open(f"{result_dir}/musique_ans_v1.0_dev_1000_pred_{avg_context_length}.jsonl",'w') as f:
        for res in result:
            f.write(json.dumps(res)+'\n')

if __name__ == "__main__":
    main()
