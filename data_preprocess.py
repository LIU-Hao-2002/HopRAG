import json
from tqdm import tqdm
import jsonlines
import os

def process_data(source_path, docs_dir, output_path):
    doc2id = {}
    
    # Open and load the source data
    with open(source_path, 'r') as f:
        data = json.load(f)
    
    # Process the entries and create text files for documents
    for temp in tqdm(data):
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
    
    # Write the data to a jsonlines file
    with jsonlines.open(output_path, mode='w') as writer:
        for result in data:
            writer.write(result)

def process_data_musique(source_path, docs_dir):
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    doc2id={}
    count=0
    with open(source_path, 'r') as f:
        data=[json.loads(line) for line in f.readlines()]
    id2txt={}
    for temp in data:
        id=temp['id']
        paragraphs=temp['paragraphs']
        for dic in paragraphs:
            idx=dic['idx']
            text=dic['paragraph_text']
            txtname=id+'_'+str(idx)
            if text not in doc2id:
                doc2id[text]=txtname
            else:
                count+=1
                txtname=doc2id[text]
            if id in id2txt:
                id2txt[id].append(txtname)
            else:
                id2txt[id]=[txtname] 
    unique_txtname=[]
    for txtnames in id2txt.values():
        unique_txtname+=txtnames
    assert len(set(unique_txtname))==len(doc2id)
    for text,new_id in doc2id.items():
        if '/' in new_id:
            new_id=new_id.replace('/','_')
        with open(docs_dir+'/'+new_id+'.txt','w') as f:
            f.write(text)
    with open(source_path.replace('.jsonl','_id2txt.json'),'w') as f:
        json.dump(id2txt,f)
    

def main_hotpot_2wiki():
    # for hotpotqa or 2wiki dataset, you can proprocess it like this:
    source_path = 'quickstart_dataset/hotpot_example.json'
    docs_dir = 'quickstart_dataset/hotpot_example_docs'
    output_path = source_path.replace('.json', '.jsonl')    
    # Call the function with the provided paths
    process_data(source_path, docs_dir, output_path)

def main_musique():
    # for musique dataset, you can proprocess it like this:
    source_path = 'quickstart_dataset/musique_example.jsonl'
    docs_dir = 'quickstart_dataset/musique_example_docs'  
    # Call the function with the provided paths
    process_data_musique(source_path, docs_dir)

if __name__ == "__main__":
    main_hotpot_2wiki()


