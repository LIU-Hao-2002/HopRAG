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

if __name__ == "__main__":
    source_path = '/path/to/dev.json'
    docs_dir = '/path/to/2wiki_docs'
    output_path = '/path/to/your/questions.jsonl'
    
    # Call the function with the provided paths
    process_data(source_path, docs_dir, output_path)
