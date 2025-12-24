
import os
import warnings
import loguru
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tool import *
from config import *
from tqdm import tqdm
import numpy as np
from neo4j import GraphDatabase
import concurrent.futures
import pickle
import pandas as pd
import json
from typing import List, Tuple, Dict, Any,Set
import time
logger = loguru.logger
class QABuilder:
    def __init__(self,done:Set[str]={},label="hotpot_example"):
        self.emb_model = load_embed_model(embed_model)  
        self.query_generator = load_language_model(query_generator_model)
        #self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password), database=neo4j_dbname, notifications_disabled_categories=neo4j_notification_filter)
        self.driver = None # if offline, we don't need to connect to neo4j; we will connect to neo4j when necessary
        self.edges=None # pending2answerable
        self.abstract2chunk=None # pseudo abstract to chunk
        self.done=done
        self.label=label # label is the type of node in neo4j

    def get_single_doc_qa(self, doc:str)->Dict[str,Tuple[str,Set,np.ndarray,Dict[str,List[Tuple[str,Set,np.ndarray]]],str]]: 
        def process_sentence(sentence_list:List[str],keywords:Set)->Tuple[Dict[str,List[Tuple[str,Set,np.ndarray]]],np.ndarray,str]:
            # each process_sentence function deals with a node, so it can be parallelized
            if type(sentence_list)==str:
                sentence_list=[sentence_list]# fix the old bug: sentence list has to be a list!!
            if len(sentence_list)==0:
                return None
            elif len(sentence_list)==1:
                temp=sentence_list[0]
            else:
                temp=','.join(sentence_list)
            #temp is the chunk in each node(str)
            sentence_embeddings=get_doc_embeds(temp, self.emb_model)
            questions_dict={}
            question_list_answerable = get_question_list(extract_template_fixed_eng, sentence_list,query_generator=self.query_generator)  
            if len(question_list_answerable)==0:
                return None 
            answerable_embeddings=get_doc_embeds(question_list_answerable, self.emb_model)
            question_list_pending = get_question_list(extract_template_pending_eng, temp,query_generator=self.query_generator)
            if len(question_list_pending)==0:
                return None 
            pending_embeddings=get_doc_embeds(question_list_pending, self.emb_model)
            questions_dict['answerable']=[(question,keywords,emb) for question,emb in zip(question_list_answerable,answerable_embeddings)]
            questions_dict['pending']=[(question,keywords,emb) for question,emb in zip(question_list_pending,pending_embeddings)]
            return questions_dict,sentence_embeddings,self.label# two types of questions, chunk embedding and label for this node;
        
        title,keywords=get_title_keywords_eng(title_template_eng,doc,query_generator=self.query_generator)
        chunks = doc.split(signal) # For Hotpot QA, split each chunk by every "\n\n", where each sentence is a chunk inside a node; for Musique, each text is a node
        #chunks:list[str]
        #sentences: list[list[str]]
        sentences=[chunk.split(',') for chunk in chunks]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_thread_num) as executor:
            futures = [executor.submit(process_sentence, sentence_list,keywords) for sentence_list in sentences] 
            results = [future.result() for future in futures] # list of tuple 
        sentences_final=[]
        results_final=[]
        for i in range(len(sentences)):
            if results[i] is not None:
                sentences_final.append(sentences[i])
                results_final.append(results[i]) # fix bug
        outcome=dict() # sentence2node
        for sentence,result in zip(sentences_final,results_final):
            if type(sentence)==list:
                if len(sentence)==1:
                    sentence=sentence[0]
                else:
                    sentence=','.join(sentence) 
            if sentence not in outcome:
                outcome[sentence]=(sentence,keywords,result[1],result[0],result[2]) # Text, keyword embedding, question dictionary, text classification
            else:
                print('duplicate sentence:',sentence)
                outcome[sentence]=(sentence,keywords,result[1],result[0],result[2])
        return outcome 
    
    def create_nodes(self,docs_dir:str='/path/to/docs')->Tuple[Dict[str,List[int]],Dict[Tuple[int,str],Dict[str,List[Tuple[str,Set,np.ndarray]]]]]:
        logger.info(f"!!! starting creating online nodes called {self.label} for docs in {docs_dir}")
        if self.driver is None:
            self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password), database=neo4j_dbname, notifications_disabled_categories=neo4j_notification_filter)
        docs_pool=os.listdir(docs_dir) 
        docid2nodes={}
        node2questiondict={}
        with self.driver.session() as session:
            for doc_id in tqdm(docs_pool,desc='create_nodes'): 
                if doc_id in self.done:
                    continue
                try:
                    nodes_id=[]
                    doc_dir=os.path.join(docs_dir,doc_id)
                    with open(doc_dir,'r') as f:
                        doc=f.read()
                        sentence2node=self.get_single_doc_qa(doc)
                        for text,tup in sentence2node.items():
                            node={'text':tup[0],'keywords':sorted(list(tup[1])),'embed':tup[2]} # Convert the keywords set to a list before passing it to the Neo4j query
                            type=self.label
                            node_id=session.run(create_entity_query.format(type=type),{'text':node['text'],'keywords':node['keywords'],'embed':node['embed']}).single()[0] # Add the attributes later when the edges are created
                            node2questiondict[(node_id,doc_id)]=tup[3]
                            nodes_id.append(node_id)
                    docid2nodes[doc_id]=nodes_id
                except Exception as e:
                    logger.info(f'error:{doc_id}——{e}')
                    time.sleep(3)
                    continue

        return docid2nodes,node2questiondict# 
    
    def create_nodes_offline(self,docs_dir:str='/path/to/docs',start_index=0,span=100)->Tuple[Dict[str,List[int]],Dict[Tuple[int,str],Dict[str,List[Tuple[str,Set,np.ndarray]]]]]:
        logger.info(f" starting creating offline nodes called {self.label} for docs in {docs_dir} from index {start_index} to {start_index+span-1}")
        docs_pool=os.listdir(docs_dir) 
        docid2nodes={}
        node2questiondict={}
        node_id=start_index*50 # assume 50 nodes for each previous doc to avoid node_id conflict
        for doc_id in tqdm(docs_pool[start_index:start_index+span],desc='create_nodes'): 
            if doc_id in self.done:
                continue
            try:
                nodes_id=[]
                doc_dir=os.path.join(docs_dir,doc_id)
                with open(doc_dir,'r') as f:
                    doc=f.read()
                    sentence2node=self.get_single_doc_qa(doc)
                    for text,tup in sentence2node.items():
                        node={'text':tup[0],'keywords':sorted(list(tup[1])),'embed':tup[2]} # Convert the keywords set to a list before passing it to the Neo4j query
                        type=self.label # since we don't need to push nodes online here, pls do not uncomment the following line
                        # node_id=session.run(create_entity_query.format(type=type),{'text':node['text'],'keywords':node['keywords'],'embed':node['embed']}).single()[0] # Add the attributes later when the edges are created
                        node_id+=1
                        node2questiondict[(node_id,doc_id)]=(node,tup[3]) # cache the node
                        nodes_id.append(node_id)
                docid2nodes[doc_id]=nodes_id
            except Exception as e:
                logger.info(f'error:{doc_id}——{e}')
                time.sleep(3)
                continue
        return docid2nodes,node2questiondict# 
    
    def create_nodes_cache(self,cache_dir:str="path/to/cache_dir")->Tuple[Dict[str,List[int]],Dict[Tuple[int,str],Dict[str,List[Tuple[str,Set,np.ndarray]]]]]:
        logger.info(f'!!! creating {self.label} nodes online from {cache_dir}')
        if self.driver is None:
            self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password), database=neo4j_dbname, notifications_disabled_categories=neo4j_notification_filter)
        with open(f'{cache_dir}/node2questiondict.pkl','rb') as f:
            old_node2questiondict=pickle.load(f) # please notice that old_node2questiondict from offline is (nodeid,docid) to nodedict
        with open(f'{cache_dir}/docid2nodes.json','r') as f:
            old_docid2nodes=json.load(f)
        new_node2questiondict={}
        new_docid2nodes={}
        cnt=0
        with self.driver.session() as session:
            for doc_id,old_node_ids in old_docid2nodes.items():
                if cnt%10==0:
                    logger.info(f'processing doc {doc_id} with {len(old_node_ids)} nodes and {cnt} docs processed so far')
                    time.sleep(1)
                cnt+=1
                nodes_id=[]# one new node for each sentence
                for old_node in old_node_ids:
                    node,questiondict=old_node2questiondict[(old_node,doc_id)]
                    type=self.label ###
                    node_id=session.run(create_entity_query.format(type=type),{'text':node['text'],'keywords':node['keywords'],'embed':node['embed']}).single()[0] # Add the attributes later when the edges are created
                    new_node2questiondict[(node_id,doc_id)]=questiondict
                    nodes_id.append(node_id)
                new_docid2nodes[doc_id]=nodes_id

        return new_docid2nodes,new_node2questiondict 
    
    def create_edge(self,node2questiondict,docid2nodes):
        # table:nodeid question_label question_id embedding question keywords
        if self.driver is None:
            self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password), database=neo4j_dbname, notifications_disabled_categories=neo4j_notification_filter)
        def get_sparse_similarity_transform(group):
            group['sparse_similarity']=sparse_similarities_result[(str(group.iloc[0]['keywords_x']),str(group.iloc[0]['keywords_y']))]
            return group
        N=len(node2questiondict)
        data=[]
        for key,value in node2questiondict.items():
            node_id,doc_id=key
            for question_label,tuplelist in value.items():
                for i,tuple in enumerate(tuplelist):
                    question,keywords,emb=tuple
                    question_id=i
                    data.append({'doc_id':doc_id,'node_id':node_id,'question_label':question_label,'question_id':question_id,'embedding':emb,'question':question,'keywords':keywords})
                    # insert into table
        del node2questiondict
        df=pd.DataFrame(data,columns=['doc_id','node_id','question_label','question_id','embedding','question','keywords'])
        del data
        sparse_similarities_result=sparse_similarities_df(df) # n**2 space complexity
        answerable_df=df[df['question_label']=='answerable']
        pending_df=df[df['question_label']=='pending']
        del df
        cartesian=pending_df.merge(answerable_df,how='cross') # x:pending y:answerable

        dense_similarity=pending_dot_answerable(pending_df,answerable_df)
        del pending_df
        cartesian['dense_similarity']=dense_similarity
        cartesian=cartesian.loc[cartesian['node_id_x']!=cartesian['node_id_y']] # Nodes cannot form self-loops, but they can connect to different sentences within the same document (i.e., different nodes)
        del dense_similarity
        cartesian=cartesian.groupby(['doc_id_x','doc_id_y']).apply(get_sparse_similarity_transform).reset_index(drop=True)#
        del sparse_similarities_result
        cartesian['similarity']=cartesian['dense_similarity']+cartesian['sparse_similarity'] # Weight
        idx=cartesian.groupby('question_x')['similarity'].idxmax() # For each follow-up question, find the most relevant answer question, which may come from the same document but different nodes, or from different documents' nodes
        cartesian1=cartesian.loc[idx] 
        cartesian2=cartesian.loc[cartesian['doc_id_x']!=cartesian['doc_id_y']] # To avoid building edges all within the same document, a fallback edge creation step ensures different documents. However, the final similarity trimming is done together with edges from the same document (the downside is that this part tends to retain fewer edges)
        del cartesian,idx
        cartesian1['keywords_both']=cartesian1.apply(lambda x:x['keywords_x'].union(x['keywords_y']),axis=1) # try cartesian1.swifter.apply for faster speed with package swifter
        self.edges=cartesian1[['node_id_x','question_y','keywords_both','embedding_x','node_id_y','similarity']] # Edges should retain those pointing to the question
        self.abstract2chunk=answerable_df.loc[~answerable_df['question'].isin(cartesian1['question_y']) & ~answerable_df['question'].isin(cartesian2['question_y'])] # No answerable questions that match any follow-up questions
        del answerable_df,cartesian1

        cartesian2 = cartesian2.sort_values(by=['question_x', 'similarity'], ascending=[True, False])
        idx = cartesian2.groupby('question_x').head(2).index # Encourage multiple hops between documents, so the value here is 2
        cartesian2=cartesian2.loc[idx]
        del idx
        cartesian2['keywords_both']=cartesian2.apply(lambda x:x['keywords_x'].union(x['keywords_y']),axis=1) # try cartesian2.swifter.apply for faster speed with package swifter

        max_edges_num=1000000000
        cartesian2=cartesian2.sort_values(by='similarity',ascending=False).drop_duplicates(subset=['node_id_x','node_id_y'],keep='first')
        cartesian2_trimmed=cartesian2.iloc[int(max_edges_num):] # Remove the dissimilar edges, then select some of them as supplements to ensure each node can be exited
        cartesian2=cartesian2.iloc[:int(max_edges_num)]
        cartesian2_trimmed=cartesian2_trimmed.loc[~cartesian2_trimmed['node_id_x'].isin(cartesian2['node_id_x'])].groupby('node_id_x').head(1) # Each node
        cartesian2=pd.concat([cartesian2,cartesian2_trimmed],ignore_index=True) # Ensure each node has at least one edge, and the edge is between documents
        self.edges=self.edges.sort_values(by='similarity',ascending=False).drop_duplicates(subset=['node_id_x','node_id_y'],keep='first')
        cartesian2=cartesian2.sort_values(by='similarity',ascending=False).drop_duplicates(subset=['node_id_x','node_id_y'],keep='first')
        inner_ratio=1/4
        self.edges=self.edges.iloc[:int(max_edges_num*inner_ratio)]
        cartesian2=cartesian2.iloc[:int(max_edges_num*(1-inner_ratio))] # Limit the total number of edges to N * np.log(N); the proportion of edges within the document is 1, and the proportion between documents is 3
        self.edges=pd.concat([self.edges,cartesian2[['node_id_x','question_y','keywords_both','embedding_x','node_id_y','similarity']]],ignore_index=True) 
        self.edges=self.edges.drop_duplicates(subset=['node_id_x','node_id_y'],keep='first')
        del cartesian2
        with self.driver.session() as session:
            for i,row in self.edges.iterrows():
                session.run(create_pending2answerable,{'id1':row['node_id_x'],'id2':row['node_id_y'],'keywords':sorted(list(row['keywords_both'])),'embed':row['embedding_x'],'answerable_question':row['question_y']})# 【】

        if len(self.abstract2chunk)==0:
            return 
        with self.driver.session() as session:
            for i,row in self.abstract2chunk.iterrows():
                temp_keywords=sorted(list(row['keywords']))
                doc_id=row['doc_id']
                abstract_id=docid2nodes[doc_id][0]
                session.run(create_abstract2answerable,{'abstract_id':abstract_id,'id2':row['node_id'],'keywords':temp_keywords,'embed':row['embedding'],'answerable_question':row['question']})



    def create_edges_musique(self,node2questiondict,docid2nodes,problems_path="/path/to/musique/musique_problems.jsonl"):
        with open(problems_path,'r') as f:
            problems=[json.loads(line) for line in f]
        id2txt=json.load(open(problems_path.replace('.jsonl','_id2txt.json'),'r')) # this file is created in process_data_musique in data_preprocess.py
        for problem in tqdm(problems,'create_edges_musique'): # 
            id=problem['id']
            if id in self.done:
                continue
            txts=id2txt[id]
            docs=[x+'.txt' for x in txts] # All the text documents corresponding to the question with this ID
            docid2nodes_={x:docid2nodes[x] for x in docs if x in docid2nodes}
            nodes=[(y,x) for x in docid2nodes_.keys() for y in docid2nodes_[x]]
            node2questiondict_={(y,x):node2questiondict[(y,x)] for (y,x) in nodes} 
            try:
                self.create_edge(node2questiondict_,docid2nodes_)
                self.done.add(id)
            except Exception as e:
                logger.info(f'{id} error {e}')
                continue

    def create_edges_hotpot(self,node2questiondict,docid2nodes,problems_path="/path/to/hotpotqa/hotpotqa_problems.jsonl"):
        with open(problems_path,'r') as f:
            problems=[json.loads(line) for line in f]
        for problem in tqdm(problems,'create_edges'): 
            id=problem['_id']
            if id in self.done:
                continue
            context=problem['context']
            docs=[x[0].replace('/','_')+'.txt' for x in context]
            docid2nodes_={x:docid2nodes[x] for x in docs if x in docid2nodes} 
            nodes=[(y,x) for x in docid2nodes_.keys() for y in docid2nodes_[x]]
            node2questiondict_={(y,x):node2questiondict[(y,x)] for (y,x) in nodes} 
            try:
                self.create_edge(node2questiondict_,docid2nodes_)
                self.done.add(id)
            except Exception as e:
                logger.info(f'{id} error {e}')
                continue # for hotpot； 

    def create_index(self): # change the xxx_name in config.py before calling
        if self.driver is None:
            self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password), database=neo4j_dbname, notifications_disabled_categories=neo4j_notification_filter)
        with self.driver.session() as session:
            index,type=node_dense_index_name,self.label
            index_cypher = create_node_dense_index_template.format(name=index, property="embed", dim=embed_dim,type=type)
            session.run(index_cypher)
            index,type=edge_dense_index_name,edge_name
            index_cypher = create_edge_dense_index_template.format(name=index,  property="embed", dim=embed_dim,type=type)
            session.run(index_cypher)
            index,type=node_sparse_index_name,self.label 
            index_cypher = create_node_sparse_index_template.format(name=index, property="keywords",type=type) # Both edges and nodes have attributes as lists during creation
            session.run(index_cypher)
            index,type=edge_sparse_index_name,edge_name
            index_cypher = create_edge_sparse_index_template.format(name=index, property="keywords",type=type) # Both edges and nodes have attributes as lists during creation
            session.run(index_cypher)            


def main_nodes(cache_dir='quickstart_dataset/cache_hotpot',docs_dir="quickstart_dataset/hotpot_example_docs",label="hotpot_test",start_index=0,span=50,original_cache_dir=None,offline=True):
    logger.info(f"starting indexing docs from {docs_dir}: from starting index {start_index} to ending index {start_index+span-1}")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    start_time=time.time()
    print('start',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    if os.path.exists(f'{cache_dir}/docid2nodes.json'):
        with open(f'{cache_dir}/docid2nodes.json','r') as f: 
            docid2nodes_old = json.load(f)
    else:
        docid2nodes_old={}
    done=set(docid2nodes_old.keys())
    builder = QABuilder(done=done,label=label)
    if original_cache_dir is None:
        if offline:
            docid2nodes,node2questiondict=builder.create_nodes_offline(docs_dir,start_index=start_index,span=span)
        else:
            docid2nodes,node2questiondict=builder.create_nodes(docs_dir)
    else:
        docid2nodes,node2questiondict=builder.create_nodes_cache(original_cache_dir)###
    print(docid2nodes)#  
    if os.path.exists(f'{cache_dir}/node2questiondict.pkl'):
        with open (f'{cache_dir}/node2questiondict.pkl','rb') as f:
            node2questiondict_old=pickle.load(f)
    else:
        node2questiondict_old={}
    node2questiondict_old.update(node2questiondict)
    with open (f'{cache_dir}/node2questiondict.pkl','wb') as f:
        pickle.dump(node2questiondict_old,f)
    del node2questiondict_old
    docid2nodes_old.update(docid2nodes)
    with open(f'{cache_dir}/docid2nodes.json','w') as f:
        json.dump(docid2nodes_old,f)
    del docid2nodes_old
    if builder.driver is not None:
        builder.driver.close()
        builder.driver=None
    end_time=time.time()
    print('time:',end_time-start_time)

def main_edges_index(cache_dir='quickstart_dataset/cache_hotpot',problems_path="quickstart_dataset/hotpot_example.jsonl",label='hotpot_test'):
    logger.info(f"!!! starting creating edges called {edge_name} for node {label} and then build index called {node_dense_index_name} and {edge_dense_index_name} and {node_sparse_index_name} and {edge_sparse_index_name}")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    start_time=time.time()
    print('start',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    if os.path.exists(f'{cache_dir}/docid2nodes.json'):
        with open(f'{cache_dir}/docid2nodes.json','r') as f: 
            docid2nodes_old = json.load(f)
    else:
        docid2nodes_old={}
    docid2nodes=docid2nodes_old
    if os.path.exists(f'{cache_dir}/edges_done.pkl'):
        with open(f'{cache_dir}/edges_done.pkl','rb') as f:
            done=pickle.load(f)
    else:
        done=set()
    builder = QABuilder(done=done,label=label)
    if os.path.exists(f'{cache_dir}/node2questiondict.pkl'):
        with open (f'{cache_dir}/node2questiondict.pkl','rb') as f:
            node2questiondict_old=pickle.load(f)
    else:
        node2questiondict_old={}
    node2questiondict=node2questiondict_old
    if "musique" in label:
        builder.create_edges_musique(node2questiondict,docid2nodes,problems_path=problems_path)
    else:
        builder.create_edges_hotpot(node2questiondict,docid2nodes,problems_path=problems_path)  
    with open(f'{cache_dir}/edges_done.pkl','wb') as f:
        pickle.dump(builder.done,f)
    end_time=time.time()
    print('time:',end_time-start_time)

    builder.create_index()
    if builder.driver is not None:
        builder.driver.close()
        builder.driver=None

if __name__ == "__main__":
    # for creating nodes, there are two ways: 1. offline and online seperate; 2. offline and online hybrid. the first one recommended.

    # 1. separate mode has two consecutive steps:
    #   (1) offline mode:first change cuda device and nodename in config.py
    # main_nodes(cache_dir='quickstart_dataset/cache_hotpot_offline',docs_dir="quickstart_dataset/hotpot_example_docs",label=node_name,
    #                start_index=0,span=12000)

    #   (2) after finishing (1), push offline cache to online neo4j; first change cuda device and nodename in config.py
    # this step will create new cache_dir (e.g. cache_hotpot_online), feel free to delete original_cache_dir after finishing online indexing
    # main_nodes(cache_dir='quickstart_dataset/cache_hotpot_online',docs_dir="quickstart_dataset/hotpot_example_docs",label=node_name,
    #                start_index=0,span=12000,original_cache_dir='quickstart_dataset/cache_hotpot_offline')  
    
    # 2. hybrid mode is an alternative way to create nodes and edges in one step:
    main_nodes(cache_dir='quickstart_dataset/cache_hotpot_online', docs_dir="quickstart_dataset/hotpot_example_docs",label=node_name,
     start_index=0,span=12000,offline=False,original_cache_dir=None)


    # for creating edges, it's much eaiser. first make sure creating nodes is finished and change dataset_name,node_name and edge_name in config.py
    main_edges_index(cache_dir="quickstart_dataset/cache_hotpot_online",
                     problems_path='quickstart_dataset/hotpot_example.jsonl',
                     label=node_name)
    

    
# nohup python HopBuilder.py >> hotpot_builder.txt &