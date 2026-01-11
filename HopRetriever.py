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
from typing import List, Tuple, Dict, Set, Union
from collections import defaultdict

class HopRetriever:
    def __init__(self,llm='gpt-4o-mini',max_hop:int=5,entry_type="edge",if_hybrid=False,if_trim=False,cache_context_path="./context_outcome.json",tol=2,mock_dense=False,mock_sparse=False,topk=10,traversal="bfs",embedding_model=embed_model,reranker=None):
        self.emb_model = load_embed_model(embedding_model)
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password), database=neo4j_dbname, notifications_disabled_categories=neo4j_notification_filter)
        self.max_hop = max_hop
        self.entry_type = entry_type
        self.cache_context_path = cache_context_path
        self.if_hybrid = if_hybrid
        self.if_trim = if_trim
        self.tol = tol
        self.mock_dense = mock_dense
        self.mock_sparse = mock_sparse
        self.reasoning_model = load_language_model(llm)
        self.topk=topk
        self.traversal=traversal
        if reranker is not None:
            model,tokenizer=load_rerank_model(reranker)
            self.rerank_model,self.rerank_tokenizer=model,tokenizer
            self.topk=topk*2

    def process_query(self,query):
        # get embedding and keywords for hybrid retrieval
        query_embedding=get_doc_embeds(query, self.emb_model)
        query_keywords=query # str
        return query_embedding, query_keywords
    
    def query_reformulation(self,query):
        chat = [] 
        chat.append({"role": "user", "content": query_reformulation_template.format(query=query)})
        outcome = get_chat_completion(chat, keys=["Subqueries"],model=self.reasoning_model)
        subqueries=outcome[0]
        return subqueries
    
    def hybrid_retrieve_edge(self,keywords:str,embedding:List,context:Dict,mock_sparse:bool=False):
        startNode_sparse=[]
        startNode_dense=[]
        with self.driver.session() as session:
            result=session.run(retrieve_edge_sparse_query.format(keywords=repr(keywords),index=repr(edge_sparse_index_name)))
            if result is None:
                return None
            for record in result: 
                startNode_sparse.append((record['endNode'],record['sparse_edge'],record['sparse_score']))
            if mock_sparse:
                return startNode_sparse
            result=session.run(retrieve_edge_dense_query.format(embedding=embedding,index=repr(edge_dense_index_name)))
            if result is None:
                return None
            for record in result:
                startNode_dense.append((record['endNode'],record['dense_edge'],record['dense_score']))

        startNode_hybrid=[(x[0],x[2]+y[2]) for x in startNode_sparse for y in startNode_dense if x[1]['question']==y[1]['question']]
        if len(startNode_hybrid)==0:
            startNode_hybrid=startNode_dense
        startNode_hybrid=[(node,score) for node,score in startNode_hybrid if node['text'] not in context] # Exclude nodes that are already in the context


        startNode_hybrid=sorted(startNode_hybrid,key=lambda x:x[1],reverse=True)
        return startNode_hybrid # List[Tuple[Dict,float]]
        
    
    def hybrid_retrieve_node(self,keywords:str,embedding:List,context:Dict,mock_sparse:bool=False):
        startNode_sparse=[]
        startNode_dense=[]
        with self.driver.session() as session:
            result=session.run(retrieve_node_sparse_query.format(keywords=repr(keywords),index=repr(node_sparse_index_name)))
            if result  is None:
                return None
            for record in result:
                startNode_sparse.append((record['sparse_node'],record['sparse_score']))
            if mock_sparse:
                return startNode_sparse
            result=session.run(retrieve_node_dense_query.format(embedding=embedding,index=repr(node_dense_index_name)))
            if result is None: 
                return None
            for record in result:
                startNode_dense.append((record['dense_node'],record['dense_score']))

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
            result=session.run(retrieve_node_dense_query.format(embedding=embedding,index=repr(node_dense_index_name)))
            if result is None:
                return None
            for record in result:
                startNode_dense.append((record['dense_node'],record['dense_score']))

        startNode_dense=sorted(startNode_dense,key=lambda x:x[1],reverse=True)
        startNode_dense=[(node,score) for node,score in startNode_dense if node['text'] not in context] # Exclude nodes already present in the context

        if len(startNode_dense)==0:
            return None
        return startNode_dense # List[Tuple[Dict,float]]
    
    def dense_retrieve_edge(self,embedding:List,context:Dict):
        startNode_dense=[]
        with self.driver.session() as session:
            result=session.run(retrieve_edge_dense_query.format(embedding=embedding,index=repr(edge_dense_index_name)))
            if result is None:
                return None
            for record in result:
                startNode_dense.append((record['endNode'],record['dense_edge'],record['dense_score']))

        startNode_dense=[(node,score) for node,edge,score in startNode_dense if node['text'] not in context] # Exclude nodes already present in the context
        return startNode_dense # List[Tuple[Dict,float]]

    def sparse_retreive_node(self,keywords:str,context:Dict):
        startNode_sparse=[]
        with self.driver.session() as session:
            result=session.run(retrieve_node_sparse_query.format(keywords=repr(keywords),index=repr(node_sparse_index_name)))
            if result  is None:
                return None
            for record in result:
                startNode_sparse.append((record['sparse_node'],record['sparse_score']))

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
    
    def get_llm_choice(self,current_node,context,query)->Union[str,Tuple[str,Dict]]:
        # get llm choice using llm_choice_query to first judge node content then out edges 
        out_questions= []
        out_nodes=[]
        with self.driver.session() as session:
            result=session.run(get_out_edge_query,{'embed':current_node['embed'],'text':current_node['text']})
            for record in result:
                if record['out_node']['text'] in context:
                    continue
                out_questions.append(record['out_edge']['question'])
                out_nodes.append(record['out_node'])

        if len(out_questions)==0:
            return 'Lack Queries'

        questions=dict(zip(range(1,len(out_questions)+1),out_questions))
        que2node=dict(zip(out_questions,out_nodes))
        chat = [] 
        chat.append({"role": "user", "content": llm_choice_query.format(node_content=current_node['text'],query=query,choices=questions)})
        outcome = get_chat_completion(chat, keys=["Decision"],model=self.reasoning_model)
        choice = outcome[0]
        if choice=="Lack Queries":
            return choice
        elif choice=="Not Needed":
            return choice
        else:
            try:
                choice=int(choice)
                if choice in questions: # return "1" or "2" or etc. in newer version of llm_choice_query
                    return "Follow-up",que2node[questions[choice]]
            except:
            # if outcome[1] is the specific question, then map back to node
                try:
                    return "Follow-up",que2node[choice]
                except:
                    try:
                        return "Follow-up",que2node[[x for x in que2node.keys() if x[-7:]==choice[-7:]][0]]
                    except:
                        print("fail extraction")
                        return "Lack Queries"

    def get_llm_hop2(self,current_node,context,query,judged_outcome,helpful_nodes):
        # get llm hop using llm_hop2_query to judge whether current node is helpful and whether to hop to the next node; return list of next nodes(dict)
        que2node={}
        hops_nodes={} # node_content:str -> node:Dict     
        # for node
        chat = [] 
        chat.append({"role": "user", "content": llm_node_choice_prompt.format(node_content=current_node['text'],query=query)})
        outcome = get_chat_completion(chat, keys=["Decision"],model=self.reasoning_model)
        choice = outcome[0]
        if choice=="helpful":
            helpful_nodes.add(current_node['text'])        
        # for edges
        with self.driver.session() as session:
            result=session.run(expand_node_edge_query,{'text':current_node['text']})# expand start node through logical relationships
            for record in result:
                if record['out_edge']['question'] in que2node:
                    continue
                que2node[record['out_edge']['question']] = record['out_node']
        edge_cnt = 0
        for question,node in que2node.items():
            edge_cnt += 1
            if node['text'] in hops_nodes:continue # already added to next hops
            if question not in judged_outcome:
                chat = [] 
                chat.append({"role": "user", "content": llm_edge_choice_prompt.format(question=question,query=query)})
                outcome = get_chat_completion(chat, keys=["Decision"],model=self.reasoning_model)
                choice = outcome[0]
                judged_outcome[question] = choice
            choice = judged_outcome[question]
            if choice=="helpful":
                hops_nodes[node['text']]=node
            if edge_cnt>=8:
                break # limit to 8 edges
        print(f"current_node has {len(que2node)} edges, uses {edge_cnt} edges and brings out {len(hops_nodes)} hops")
        judged_outcome[current_node['text']]=list(hops_nodes.values())
        return judged_outcome,helpful_nodes
    
    def find_next_node(self,current_node:Dict,context:Dict,query:str,node2score:Dict[str,float],query_embedding):
        # First, exclude the recalled results that are already in the context to avoid duplicates
        llm_choice=self.get_llm_choice(current_node,context,query)
        next_node_sim = None
        if llm_choice == 'Not Needed': # 
            return current_node,-1
        elif llm_choice=='Lack Queries': # The node is indeed necessary, but there is no suitable Follow-up question
            next_node = "Lack Queries" 
        elif len(llm_choice)==2:
            if llm_choice[0]=='Follow-up': # Both the node and the outlier node are necessary
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
            if next_node=="Lack Queries":
                # The current node is necessary, but there is no suitable Follow-up question. Start from the next starting point and restart the search, ending the current walk.
                return context,True
            if not next_node:  # Unable to find the next node, end the current walk
                return context, False
            if node_sim == -1:  # Either this information is not needed to answer the question, or there is no next node
                context[next_node['text']] -= 0.2  # Penalty: the next_node is a local current node that cannot be skipped, not the next node, since there is no next one
                return context, False
            context[next_node['text']] = node_sim  # Nodes that cannot be exited might be irrelevant to the question. Set similarity to -1 but still add to the context to indicate it has been visited; otherwise, it will cause an infinite loop
            current_node = next_node
        return context,None
    
    def topk_filter(self,sim_dict:Dict[str,float])->Tuple[List[str],List[float]]:
        logger.info(f"Total {len(sim_dict)} nodes after traversal, filtering top {self.topk} nodes as final context")
        outcome = sorted(sim_dict.items(),key=lambda x:x[1],reverse=True)
        final_context = []
        final_score = []
        for i in range(min(self.topk,len(outcome))):
            final_context.append(outcome[i][0])
            final_score.append(outcome[i][1])
        return final_context, final_score
    
    def search_docs_mock(self,query_embedding,query_keywords,topk)->Tuple[List[str],List[float]]:
        scores=[]
        context=[]
        if self.mock_dense:
            if self.entry_type=="node":
                start_node=self.dense_retrieve_node(query_embedding, {})
            else:
                start_node=self.dense_retrieve_edge(query_embedding, {})
            if not start_node:
                start_node=[]
            for i in range(topk):
                context.append(start_node[i][0]['text'])
                scores.append(start_node[i][1])
            return context,scores
        elif self.mock_sparse:
            if self.entry_type=="node":
                start_node=self.hybrid_retrieve_node(query_keywords,[],{},mock_sparse=True)
            else:
                start_node=self.hybrid_retrieve_edge(query_keywords,[],{},mock_sparse=True)
            if not start_node:
                start_node=[]
            for i in range(topk):
                context.append(start_node[i][0]['text'])
                scores.append(start_node[i][1])
            return context,scores
        else:
            if self.if_hybrid:
                if self.entry_type=="node":
                    start_node=self.hybrid_retrieve_node(query_keywords,query_embedding,{},mock_sparse=False)
                elif self.entry_type=="edge":
                    start_node=self.hybrid_retrieve_edge(query_keywords,query_embedding,{},mock_sparse=False)
                else:
                    raise ValueError("entry_type must be 'node' or 'edge'")
            else:
                if self.entry_type=="node":
                    start_node=self.dense_retrieve_node(query_embedding, {})
                elif self.entry_type=="edge":
                    start_node=self.dense_retrieve_edge(query_embedding, {})
                else:
                    raise ValueError("entry_type must be 'node' or 'edge'")
            return None, start_node
    
    def search_docs_dfs(self,query:str)->Tuple[List[str],List[float]]:
        # In DFS, topk = self.max_hop + 1
        query_embedding, query_keywords = self.process_query(query)
        mock_result=self.search_docs_mock(query_embedding,query_keywords,self.max_hop+1)
        if mock_result[0] is not None:
            return mock_result
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
        final_context = []
        final_score = []
        if self.if_trim is not False:
            if self.if_trim==True:
                node_sims = list(context.values())
                mean_sim = np.mean(node_sims)
                for key,value in context.items():
                    if value>=mean_sim:
                        final_context.append(key)
                        final_score.append(value)
            else:
                keep_num=int(0.75*len(context))+1
                for i in range(keep_num):
                    key=list(context.keys())[i]
                    final_context.append(key)
                    final_score.append(context[key])
        else:
            # Ensure the context order is consistent with the similarity order
            final_context, final_score = list(context.keys()), list(context.values())
        return final_context[:self.max_hop+1], final_score[:self.max_hop+1]
        
    def search_docs_bfs(self,query:str)->Tuple[List[str],List[str]]:
        query_embedding, query_keywords = self.process_query(query)
        mock_result=self.search_docs_mock(query_embedding,query_keywords,self.topk)
        if mock_result[0] is not None:
            return mock_result
        else:
            start_node_dense=mock_result[1]
        if self.traversal=='bfs':
            queue=[x[0]['text'] for x in start_node_dense][:self.topk] # for judge
        elif self.traversal=='bfs_sim_node':
            queue=[(x[0]['text'],x[0]['embed']) for x in start_node_dense][:self.topk] # for judge_sim
        else:
            raise ValueError("traversal type must be 'bfs' or 'bfs_sim_node'")
        count=0
        judged_outcome={}
        outcome=[]
        with self.driver.session() as session:
            while count<self.max_hop:
                queue=queue[:2*self.topk] # keep the queue size within topk
                count+=1
                api_call_time=0
                queue_irrelevant=[]
                for i in range(len(queue)):
                    if self.traversal=='bfs':
                        node_content=queue.pop(0)
                    elif self.traversal=='bfs_sim_node':
                        node_content,node_emb=queue.pop(0)
                    if node_content not in judged_outcome:
                        api_call_time+=1
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
                print(f"current count:{count},calling times:{api_call_time}")
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
                #if len(helpful)>=5:
                #    break
                if count<self.max_hop and len(queue)<5: # If there aren’t enough hops or the queue is empty, refill the queue
                    logger.info(f"{len(helpful)} helpful nodes found, {len(relevant)} relevant nodes found, {len(irrelevant)} irrelevant nodes found, {len(queue)} nodes in queue")
                    queue+=queue_irrelevant
        final_context = []
        final_score = []
        for i in range(self.topk):
            final_context.append(outcome[i])
            final_score.append(judged_outcome[outcome[i]])
        return final_context, final_score   

    def search_docs_bfs_hop2(self,query:str)->Tuple[List[str],List[float]]:
        query_embedding, query_keywords = self.process_query(query)
        mock_result=self.search_docs_mock(query_embedding,query_keywords,self.topk)
        if mock_result[0] is not None:
            return mock_result
        else:
            start_node_dense=mock_result[1]
        queue=[x[0] for x in start_node_dense][:2*self.topk] # initial queue of dict, each dict with keys: text, embed, keywords
        count=0
        judged_outcome={}# question:str -> label:str;node_content:str -> List[Dict] # list of nodes to hop to
        helpful_nodes=set()
        # initialize visit count based on initial position in queue
        visit_counter = {queue[i]['text']:1 for i in range(len(queue))}  # VIP:set to 1 to prioritize llm hop judges!!!
        node2emb = {}  # node_content:str -> emb:List[float]
        node2keyword = {}  # node_content:str -> keywords:Set[str]
        outcome=[]
        with self.driver.session() as session:
            while count<self.max_hop:
                count+=1
                api_call_time=0
                real_next_hop_node={}
                for current_node in queue[:2*self.topk]:
                    node_content = current_node['text']
                    if node_content not in judged_outcome:
                        # not judged yet
                        api_call_time+=1
                        # increment visit count for helpful nodes
                        judged_outcome,helpful_nodes=self.get_llm_hop2(current_node,{},query,judged_outcome,helpful_nodes)
                    if node_content not in node2emb:
                        node2emb[node_content]=current_node['embed']
                    if node_content not in node2keyword:
                        node2keyword[node_content]=set(current_node['keywords'])
                    
                    if node_content not in helpful_nodes:
                        if node_content not in visit_counter:
                            visit_counter[node_content] = -1
                        else:
                            pass # for unhelpful nodes when 2nd visit
                    else:
                        if node_content not in visit_counter:
                            visit_counter[node_content] = 1
                        else:
                            visit_counter[node_content] += 1 
                    next_node_list=judged_outcome[node_content]
                    for node in next_node_list:
                        if node['text'] not in real_next_hop_node:
                            real_next_hop_node[node['text']]=node
                print(f"current count:{count}, new nodes count: {len(real_next_hop_node)}, judging nodes:{api_call_time}")
                if api_call_time==0: # no more new nodes mean no more hops
                    break
                queue=list(real_next_hop_node.values()) # add to queue, but different next_node_list from for loop share common nodes!
        embeds = []
        sparse_sims = []
        for node,value in judged_outcome.items():
            if isinstance(value,str): # filter out question 
                continue
            count = visit_counter[node]
            emb = node2emb[node]
            outcome.append(node)
            embeds.append(np.array(emb))
            # sparse keyword
            sparse_sim = sparse_similarity(node2keyword[node], query_keywords)+1e-6
            sparse_sim += int(count>0)*count  ### move helpful nodes to the top
            sparse_sims.append(sparse_sim)
        embeds = np.array(embeds)
        query_embedding = np.array(query_embedding)
        sims = np.dot(embeds,query_embedding)/(np.linalg.norm(embeds,axis=1)*np.linalg.norm(query_embedding))
        sims = sims.tolist()
        # hybrid sims
        # print(visit_counter)
        print("helpful nodes:")
        print(helpful_nodes)
        for i in range(len(sims)):
            if sparse_sims[i]>=1: # helpful nodes
                count = int(sparse_sims[i])
                real_sparse_sim = sparse_sims[i]-count
                sims[i] = (0.5*sims[i]+0.5*real_sparse_sim)*count
            else:
                sims[i] = 0.5*sims[i]+0.5*sparse_sims[i]
        sim_dict = {outcome[i]:sims[i] for i in range(len(outcome))}
        final_context, final_score = self.topk_filter(sim_dict)
        return final_context, final_score   
    
    def search_docs_bfs_node(self,query:str)->Tuple[List[str],List[float]]:
        query_embedding, query_keywords = self.process_query(query)
        mock_result=self.search_docs_mock(query_embedding,query_keywords,self.topk)
        if mock_result[0] is not None:
            return mock_result
        else:
            start_node_dense=mock_result[1]
        queue=[x[0] for x in start_node_dense][:self.topk] # initial queue of dict, each dict with keys: text, embed, keywords
        count=0
        judged_outcome={}# node_content:str -> label:str
        helpful_nodes=set()
        # initialize visit count based on initial position in queue
        visit_counter = {queue[i]['text']:len(queue)-i for i in range(len(queue))}  # node_content:str -> visit_count:int
        node2emb = {}  # node_content:str -> emb:List[float]
        node2keyword = {}  # node_content:str -> keywords:Set[str]
        outcome=[]
        with self.driver.session() as session:
            while count<self.max_hop:
                queue=queue[:2*self.topk] # keep the queue size 
                count+=1
                api_call_time=0
                real_next_hop_node={}
                for current_node in queue:
                    node_content = current_node['text']
                    if node_content not in judged_outcome:
                        # not judged yet
                        api_call_time+=1
                        # increment visit count for helpful nodes
                        judged_outcome=self.judge(node_content,judged_outcome,query,llm_node_choice_prompt)
                    label=judged_outcome[node_content]
                    if label == "helpful":
                        helpful_nodes.add(node_content)
                        # increment visit count for helpful nodes
                        if node_content not in visit_counter:
                            visit_counter[node_content] = 1
                        else:
                            visit_counter[node_content] += 1
                    elif label == "helpless":
                        if node_content not in visit_counter:
                            visit_counter[node_content] = -1
                        else:
                            visit_counter[node_content] -= 1
                    else:
                        print(f"Unexpected LLM choice output. label: {label}") # should never arrive here
                        continue
                    if node_content not in node2emb:
                        node2emb[node_content]=current_node['embed']
                    if node_content not in node2keyword:
                        node2keyword[node_content]=set(current_node['keywords'])
                    result=session.run(expand_logic_query,{'text':node_content})# expand start node through logical relationships
                    for record in result:
                        new_node=record['logic_node']
                        real_next_hop_node[new_node['text']]=new_node
                print(f"current count:{count},calling times:{api_call_time}, new nodes count: {len(real_next_hop_node)}")
                if api_call_time==0: # no more new nodes mean no more hops
                    break
                queue=list(real_next_hop_node.values())
        embeds = []
        sparse_sims = []
        for node,label in judged_outcome.items():
            count = visit_counter[node]
            emb = node2emb[node]
            outcome.append(node)
            embeds.append(np.array(emb))
            # sparse keyword
            sparse_sim = sparse_similarity(node2keyword[node], query_keywords)
            sparse_sim += int(label=="helpful")*count  ### move helpful nodes to the top
            sparse_sims.append(sparse_sim)
        embeds = np.array(embeds)
        query_embedding = np.array(query_embedding)
        sims = np.dot(embeds,query_embedding)/(np.linalg.norm(embeds,axis=1)*np.linalg.norm(query_embedding))
        sims = sims.tolist()
        # hybrid sims
        # print(visit_counter)
        print("helpful nodes:\n",helpful_nodes)
        sims = [0.5*sims[i]+0.5*sparse_sims[i] for i in range(len(sims))]
        sim_dict = {outcome[i]:sims[i] for i in range(len(outcome))}
        final_context, final_score = self.topk_filter(sim_dict)
        return final_context, final_score   
    
    def judge(self,node_content:str,judged_outcome:Dict[str,str],query:str,llm_choice_query_chunk_prompt=llm_choice_query_chunk)->Dict[str,str]:
        if node_content in judged_outcome:return judged_outcome
        chat = [] 
        chat.append({"role": "user", "content": llm_choice_query_chunk_prompt.format(node_content=node_content,query=query)})
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
    
    def search_docs(self,query:str)->Tuple[List[str],List[float]]:
        if self.traversal=='dfs':
            return self.search_docs_dfs(query)
        elif self.traversal in ['bfs','bfs_sim_node']:
            return self.search_docs_bfs(query)
        elif self.traversal=="bfs_node":
            return self.search_docs_bfs_node(query)
        elif self.traversal=="bfs_hop2":
            return self.search_docs_bfs_hop2(query)
        else:
            raise ValueError("traversal type must be 'dfs' or 'bfs' or 'bfs_sim_node' or 'bfs_node' or 'bfs_hop2'")
    
    def search_docs_rerank(self,query:str)->Tuple[List[str],List[float]]:
        context,_ = self.search_docs(query)
        pairs=[]
        for passage in context:
            pair=[query,passage]
            pairs.append(pair)
        with torch.no_grad():
            inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.rerank_model.device)
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = scores.cpu().numpy()
        sorted_indices = np.argsort(scores)[::-1] # Sort in descending order
        sorted_indices = sorted_indices[:self.topk//2]
        sorted_context = [context[i] for i in sorted_indices]
        scores = [float(scores[i]) for i in sorted_indices]
        return sorted_context, scores # when using reranker topk gets doubled when init, so here it should be reduced by half
                       
if __name__ == "__main__":
    query="Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?"
    retriever = HopRetriever(llm=traversal_model,max_hop = 4, entry_type="node",if_trim=False,if_hybrid=True,
                             tol=30,topk=20,traversal='bfs_hop2',mock_dense=False,reranker=None)# no reranking then reranker=None; else: reranker=reranker
    context,scores = retriever.search_docs(query)
    print(context)
    print(scores)
    if retriever.driver is not None:
        retriever.driver.close()
        retriever.driver=None 