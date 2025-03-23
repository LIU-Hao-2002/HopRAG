import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tool import *
from config import *
import numpy as np
from neo4j import GraphDatabase
import time
from loguru import logger
from typing import List, Tuple, Dict, Set

class HopRetriever:
    def __init__(self,llm='gpt-4o-mini',max_hop:int=5,entry_type="edge",if_hybrid=False,if_trim=False,cache_context_path="./context_outcome.json",tol=2,mock_dense=False,label="hotpot_example_",mock_sparse=False,topk=10,traversal="bfs"):
        self.emb_model = load_embed_model(embed_model)
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password), database=neo4j_dbname, notifications_disabled_categories=neo4j_notification_filter)
        self.max_hop = max_hop
        self.entry_type = entry_type
        self.cache_context_path = cache_context_path
        self.if_hybrid = if_hybrid
        self.if_trim = if_trim
        self.tol = tol
        self.mock_dense = mock_dense
        self.mock_sparse = mock_sparse
        self.label = label # Determines from which index to retrieve, as different datasets have different indexes based on node type and edge type
        self.reasoning_model = llm
        self.topk=topk
        self.traversal=traversal

    def process_query(self,query):
        # get embedding and keywords for hybrid retrieval
        query_embedding=get_doc_embeds(query, self.emb_model)
        query_keywords = set(get_ner_eng(query))
        query_keywords=' '.join(sorted(list(query_keywords))) # str
        return query_embedding, query_keywords
    
    def hybrid_retrieve_edge(self,keywords:str,embedding:List,context:Dict):
        startNode_sparse=[]
        startNode_dense=[]
        with self.driver.session() as session:
            result=session.run(retrieve_edge_sparse_query.format(keywords=repr(keywords),index=repr(f'{self.label}edge_sparse_index')))
            if result is None:
                return None
            for record in result: 
                startNode_sparse.append((record['endNode'],record['sparse_edge'],record['sparse_score']))
            result=session.run(retrieve_edge_dense_query.format(embedding=embedding,index=repr(f'{self.label}edge_dense_index')))
            if result is None:
                return None
            for record in result:
                startNode_dense.append((record['endNode'],record['dense_edge'],record['dense_score']))
        self.driver.close()
        startNode_hybrid=[(x[0],x[2]+y[2]) for x in startNode_sparse for y in startNode_dense if x[1]['question']==y[1]['question']]
        if len(startNode_hybrid)==0:
            startNode_hybrid=startNode_dense
        startNode_hybrid=[(node,score) for node,score in startNode_hybrid if node['text'] not in context] # Exclude nodes that are already in the context


        startNode_hybrid=sorted(startNode_hybrid,key=lambda x:x[1],reverse=True)
        return startNode_hybrid # List[Tuple[Dict,float]]
        
    
    def hybrid_retrieve_node(self,keywords:str,embedding:List,context:Dict):
        startNode_sparse=[]
        startNode_dense=[]
        with self.driver.session() as session:
            result=session.run(retrieve_node_sparse_query.format(keywords=repr(keywords),index=repr(f'{self.label}node_sparse_index')))
            if result  is None:
                return None
            for record in result:
                startNode_sparse.append((record['sparse_node'],record['sparse_score']))
            result=session.run(retrieve_node_dense_query.format(embedding=embedding,index=repr(f'{self.label}node_dense_index')))
            if result is None: 
                return None
            for record in result:
                startNode_dense.append((record['dense_node'],record['dense_score']))
        self.driver.close()
        startNode_dense=sorted(startNode_dense,key=lambda x:x[1],reverse=True)
        startNode_hybrid=[(x[0],y[1]) for x in startNode_sparse for y in startNode_dense if x[0]['text']==y[0]['text']] # Hybrid is reflected in taking the intersection of dense and sparse results, but the internal score remains dense
        if len(startNode_hybrid)<self.max_hop:
            startNode_hybrid=startNode_dense
        startNode_hybrid=[(node,score_dense) for node,score_dense in startNode_hybrid if node['text'] not in context] # Exclude nodes that are already in the context

        startNode_hybrid=sorted(startNode_hybrid,key=lambda x:x[1],reverse=True)
        return startNode_hybrid # List[Tuple[Dict,float]]
    
    def dense_retrieve_node(self ,embedding:List,context:Dict):
        startNode_dense=[]
        with self.driver.session() as session:
            result=session.run(retrieve_node_dense_query.format(embedding=embedding,index=repr(f'{self.label}node_dense_index')))
            if result is None:
                return None
            for record in result:
                startNode_dense.append((record['dense_node'],record['dense_score']))
        self.driver.close()
        startNode_dense=sorted(startNode_dense,key=lambda x:x[1],reverse=True)
        startNode_dense=[(node,score) for node,score in startNode_dense if node['text'] not in context] # Exclude nodes already present in the context

        if len(startNode_dense)==0:
            return None
        return startNode_dense # List[Tuple[Dict,float]]
    
    def dense_retrieve_edge(self,embedding:List,context:Dict):
        startNode_dense=[]
        with self.driver.session() as session:
            result=session.run(retrieve_edge_dense_query.format(embedding=embedding,index=repr(f'{self.label}edge_dense_index')))
            if result is None:
                return None
            for record in result:
                startNode_dense.append((record['endNode'],record['dense_edge'],record['dense_score']))
        self.driver.close()
        startNode_dense=[(node,score) for node,edge,score in startNode_dense if node['text'] not in context] # Exclude nodes already present in the context
        return startNode_dense # List[Tuple[Dict,float]]

    def sparse_retreive_node(self,keywords:str,context:Dict):
        startNode_sparse=[]
        with self.driver.session() as session:
            result=session.run(retrieve_node_sparse_query.format(keywords=repr(keywords),index=repr(f'{self.label}node_sparse_index')))
            if result  is None:
                return None
            for record in result:
                startNode_sparse.append((record['sparse_node'],record['sparse_score']))
        self.driver.close()
        startNode_sparse=sorted(startNode_sparse,key=lambda x:x[1],reverse=True)
        startNode_sparse=[(node,score) for node,score in startNode_sparse if node['text'] not in context] # Exclude nodes already present in the context
        if len(startNode_sparse)==0:
            return None
        return startNode_sparse # List[Tuple[Dict,float]]
    
    def find_entry_node(self,query_embedding, query_keywords,context:Dict):
        # During the entry node search phase, precompute the edges and node rankings for the current query to facilitate context trimming

        retrieve_node=False if self.entry_type=='edge' else True
        total_score=0
        if not retrieve_node: # recommended to match nodes first
            result=self.hybrid_retrieve_edge(query_keywords,query_embedding,context)
            if not result:
                retrieve_node=True
            else:
                entry_node = result[0][0]
                total_score = result[0][1]
        if retrieve_node or total_score<=1: # If edge matching is poor, switch to node matching; the threshold can be increased
            if self.if_hybrid:
                result=self.hybrid_retrieve_node(query_keywords,query_embedding,context)
            else:
                result=self.dense_retrieve_node(query_embedding,context)
            if not result:
                return None,[]
            entry_node = result[0][0]
            total_score = result[0][1]
        node2score = {x[0]['text']:x[1] for x in result} # Dict[str,float]
        return entry_node, node2score # Return the similarity of the recalled nodes as well
    
    def get_llm_choice(self,current_node,context,query):
        time.sleep(1)
        out_questions= []
        out_nodes=[]
        with self.driver.session() as session:
            result=session.run(get_out_edge_query,{'embed':current_node['embed'],'text':current_node['text']})
            for record in result:
                if record['out_node']['text'] in context:
                    continue
                out_questions.append(record['out_edge']['question'])
                out_nodes.append(record['out_node'])
        self.driver.close()
        if len(out_questions)==0:
            return 'Lacks appropriate follow-up questions'

        questions=dict(zip(range(1,len(out_questions)+1),out_questions))
        que2node=dict(zip(out_questions,out_nodes))
        chat = [] 
        chat.append({"role": "user", "content": llm_choice_query.format(node_content=current_node['text'],query=query,choices=questions)})
        outcome = get_chat_completion(chat, keys=["Decision","Follow-up"],model=self.reasoning_model)
        choice = outcome[0]
        if choice=="Lacks appropriate follow-up questions":
            return 'Lacks appropriate follow-up questions'
        elif choice=="This information is not needed to answer the question":
            return 'This information is not needed to answer the question'
        elif choice=="Choose a follow-up question":
            if type(outcome[1])==str:
                try:
                    return "Choose a follow-up question",que2node[outcome[1]]
                except:
                    try:
                        return "Choose a follow-up question",que2node[[x for x in que2node.keys() if x[-7:]==outcome[1][-7:]][0]]
                    except:
                        return "Lacks appropriate follow-up questions"
            elif type(outcome[1])==dict:
                temp=list(outcome[1].values())[0]
                try:
                    return "Choose a follow-up question",que2node[temp]
                except:
                    return "Choose a follow-up question",que2node[[x for x in que2node.keys() if x[-10:]==temp[-10:]][0]]
            else:
                return "Lacks appropriate follow-up questions"
        else:
            return "Lacks appropriate follow-up questions"

    def get_choice(self,current_node,context,query,query_embedding):
        out_nodes=[]
        out_question_embedding=[]
        with self.driver.session() as session:
            result=session.run(get_out_edge_query,{'embed':current_node['embed'],'text':current_node['text']})
            for record in result:
                if record['out_node']['text'] in context:
                    continue
                out_nodes.append(record['out_node'])
                out_question_embedding.append(record['out_edge']['embed'])
        self.driver.close()
        if len(out_question_embedding)==0:
            return 'Lacks appropriate follow-up questions'

        out_question_embedding=np.array(out_question_embedding)
        query_embedding=np.array(query_embedding)
        sim=np.dot(out_question_embedding,query_embedding)/(np.linalg.norm(out_question_embedding,axis=1)*np.linalg.norm(query_embedding))# shape (n,)
        max_sim=np.max(sim)
        if max_sim<0.7:
            return "Lacks appropriate follow-up questions"
        max_place=np.argmax(sim)
        return "Choose a follow-up question",out_nodes[max_place]
    
    def find_next_node(self,current_node:Dict,context:Dict,query:str,node2score:Dict[str,float],query_embedding):
        # First, exclude the recalled results that are already in the context to avoid duplicates
        llm_choice=self.get_llm_choice(current_node,context,query)
        next_node_sim = None
        if llm_choice == 'This information is not needed to answer the question': # 
            return current_node,-1
        elif llm_choice=='Lacks appropriate follow-up questions': # The node is indeed necessary, but there is no suitable follow-up question
            next_node = "Lacks appropriate follow-up questions" 
        elif len(llm_choice)==2:
            if llm_choice[0]=='Choose a follow-up question': # Both the node and the outlier node are necessary
                next_node = llm_choice[1] # dict
                current_node_sim=context[current_node['text']] if current_node['text'] in context.keys() else 0.8
                next_node_sim = current_node_sim if next_node['text'] not in node2score.keys() else node2score[next_node['text']] # # Not in the top similarity rankings, but selected by the LLM
            else:
                next_node = None
        else:
            next_node = None
        return next_node , next_node_sim
    
    def random_walk(self,current_node:Dict,query,context:Set,node2score:Dict,query_embedding):
        '''DFS Random Walk'''
        while len(context)<self.max_hop+1: # In DFS, topk = self.max_hop + 1
            next_node , node_sim = self.find_next_node(current_node,context,query,node2score,query_embedding)
            if next_node=="Lacks appropriate follow-up questions":
                # The current node is necessary, but there is no suitable follow-up question. Start from the next starting point and restart the search, ending the current walk.
                return context,True
            if not next_node:  # Unable to find the next node, end the current walk
                return context, False
            if node_sim == -1:  # Either this information is not needed to answer the question, or there is no next node
                context[next_node['text']] -= 0.2  # Penalty: the next_node is a local current node that cannot be skipped, not the next node, since there is no next one
                return context, False
            context[next_node['text']] = node_sim  # Nodes that cannot be exited might be irrelevant to the question. Set similarity to -1 but still add to the context to indicate it has been visited; otherwise, it will cause an infinite loop
            current_node = next_node
        return context,None
    
    def search_docs_dfs(self,query:str)->List[str]: # In DFS, topk = self.max_hop + 1
        query_embedding, query_keywords = self.process_query(query)
        if self.mock_dense:
            start_node=self.dense_retrieve_node(query_embedding, {})
            if not start_node:
                start_node=[]
            start_node=[x[0]['text'] for x in start_node][:self.max_hop+1]
            return start_node
        if self.mock_sparse:
            start_node=self.sparse_retreive_node(query_keywords,{})
            if not start_node:
                start_node=[]
            start_node=[x[0]['text'] for x in start_node][:self.max_hop+1]
            return start_node
        context={} # Clear the context and restart the search for each query
        flags=[]
        while len(context)<self.max_hop+1: 
            entry_node, node2score = self.find_entry_node(query_embedding, query_keywords,context) # node2score will decrease as the context grows
            if not entry_node or len(node2score)==0:
                break
            context[entry_node['text']] = node2score[entry_node['text']]
            context,flag = self.random_walk(entry_node,query,context,node2score,query_embedding)
            flags.append(flag)
            if len(flags)>=self.tol and flags [-self.tol:] == [True]*self.tol or flags[-self.tol:]==[False]*self.tol: # If the walk from the starting point ends consecutively for tol times
                break
        context=dict(sorted(context.items(),key=lambda x:x[1],reverse=True))
        if self.if_trim is not False:
            if self.if_trim==True:
                node_sims = list(context.values())
                mean_sim = np.mean(node_sims)
                context = [key for key,value in context.items() if value>=mean_sim]
            else:
                keep_num=int(0.75*len(context))+1
                context = [key for key,value in context.items()][:keep_num]
        else:
            context = [key for key,value in context.items()] # Ensure the context order is consistent with the similarity order
        return context
        
    def search_docs_bfs(self,query:str)->List[str]:
        query_embedding, query_keywords = self.process_query(query)
        if self.mock_sparse:
            start_node=self.sparse_retreive_node(query_keywords,{})
            if not start_node:
                start_node=[]
            start_node=[x[0]['text'] for x in start_node][:self.topk]
            return start_node
        start_node_dense=self.dense_retrieve_node(query_embedding, {})
        if self.mock_dense:
            if not start_node_dense:
                start_node_dense=[]
            start_node_dense=[x[0]['text'] for x in start_node_dense][:self.topk]
            return start_node_dense
        if self.traversal=='bfs':
            queue=[x[0]['text'] for x in start_node_dense][:20] # for judge
        elif self.traversal=='bfs_sim_node':
            queue=[(x[0]['text'],x[0]['embed']) for x in start_node_dense][:20] # for judge_sim
        else:
            raise ValueError("traversal type must be 'bfs' or 'bfs_sim_node'")
        count=0
        judged_outcome={}
        outcome=[]
        with self.driver.session() as session:
            while count<self.max_hop:
                queue=queue[:20]#
                count+=1
                print(f"current count:{count},len(queue):{len(queue)}")
                queue_irrelevant=[]
                for i in range(len(queue)):
                    if self.traversal=='bfs':
                        node_content=queue.pop(0)
                    elif self.traversal=='bfs_sim_node':
                        node_content,node_emb=queue.pop(0)
                    if node_content not in judged_outcome:
                        if self.traversal=='bfs':
                            judged_outcome=self.judge(node_content,judged_outcome,query)
                        elif self.traversal=='bfs_sim_node':
                            judged_outcome=self.judge_sim_node(node_content,node_emb,query_embedding,judged_outcome)
                    label=judged_outcome[node_content]
                    result=session.run(expand_logic_query,{'text':node_content})# expand start node through logical relationships
                    for record in result:
                        new_text=record['logic_node']['text']
                        if self.traversal=='bfs_sim_node':
                            new_text=(new_text,record['logic_node']['embed'])
                        if label=="Completely Irrelevant":
                            queue_irrelevant.append(new_text)
                        else:
                            queue.append(new_text) # Neighbors of completely Irrelevant nodes won't be directly added to the queue, they have lower priority, unless the queue is empty and needs to be filled

                helpful=[]
                relevant=[]
                irrelevant=[]
                for node,label in judged_outcome.items():
                    if label=='Relevant and Necessary':
                        helpful.append(node)
                    elif label=='Indirectly Relevant':
                        relevant.append(node)
                    else:
                        irrelevant.append(node)
                outcome=helpful+relevant+irrelevant
                if len(helpful)>=5:
                    break
                if count<self.max_hop and len(queue)<5: # If there aren’t enough hops or the queue is empty, refill the queue
                    logger.info(f"{len(helpful)} helpful nodes found, {len(relevant)} relevant nodes found, {len(irrelevant)} irrelevant nodes found, {len(queue)} nodes in queue")
                    queue+=queue_irrelevant
        self.driver.close()
        return outcome[:self.topk]
    
    def judge(self,node_content:str,judged_outcome:Dict[str,str],query:str)->Dict[str,str]:
        if node_content in judged_outcome:return judged_outcome
        chat = [] 
        chat.append({"role": "user", "content": llm_choice_query_chunk.format(node_content=node_content,query=query)})
        outcome = get_chat_completion(chat, keys=["Decision"],model=self.reasoning_model)
        choice=outcome[0]
        judged_outcome[node_content]=choice
        return judged_outcome
    
    def judge_sim_node(self,node_content:str,node_emb:List[float],query_emb:List[float],judged_outcome:Dict[str,str])->Dict[str,str]:
        sim=np.dot(node_emb,query_emb)/(np.linalg.norm(node_emb)*np.linalg.norm(query_emb))
        if sim>0.7:
            label = 'Relevant and Necessary'
        elif sim>0.6:
            label = 'Indirectly Relevant'
        else:
            label= "Completely Irrelevant"
        judged_outcome[node_content]=label
        return judged_outcome
    
    def search_docs(self,query:str)->List[str]:
        if self.traversal=='dfs':
            return self.search_docs_dfs(query)
        elif self.traversal in ['bfs','bfs_sim_node']:
            return self.search_docs_bfs(query)
        else:
            raise ValueError("traversal type must be 'dfs' or 'bfs' or 'bfs_sim_node' ")
        
if __name__ == "__main__":
    query="Were Scott Derrickson and Ed Wood of the same nationality?"
    retriever = HopRetriever(max_hop = 4, entry_type="node",if_trim=False,if_hybrid=False,tol=30,label='hotpot_example_',topk=20,traversal='bfs',mock_dense=False)
    context = retriever.search_docs(query)
    print(context)