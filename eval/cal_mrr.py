import json
import sys
import os
not_recalled=0
result_path = sys.argv[1]
gt_path = sys.argv[2]
gt = {}
if "musique" in result_path:
    result={}
    with open(result_path, 'r') as f:
        for line in f:
            temp = json.loads(line)
            result[temp['id']] = temp['predicted_support_idxs']
    with open(gt_path, 'r') as f:
        for line in f:
            temp = json.loads(line)
            gt[temp['id']] = []
            for para in temp['paragraphs']:
                if para['is_supporting']:
                    gt[temp['id']].append(para['idx'])
else:
    result = json.load(open(result_path, 'rb'))['sp']  # id2list
    with open(gt_path, 'r') as f:
        for line in f:
            temp = json.loads(line)
            gt[temp['_id']] = temp['supporting_facts']

case_summary = {}
individual_mrr = {}  

for _id in result:
    case_summary[_id] = []
    case_result = {str(doc_id): rank for rank, doc_id in enumerate(result[_id], 1)}
    
    current_mrr = 0.0  
    first_relevant_rank = None  

    for sp in gt[_id]:
        sp_str = str(sp)
        if sp_str not in case_result:
            not_recalled+=1
            case_summary[_id].append((sp_str, -1))
        else:
            doc_rank = case_result[sp_str]
            case_summary[_id].append((sp_str, doc_rank))
            if first_relevant_rank is None or doc_rank < first_relevant_rank:
                first_relevant_rank = doc_rank


    if first_relevant_rank is not None:
        current_mrr = 1.0 / first_relevant_rank
    individual_mrr[_id] = current_mrr
    # print(f"ID: {_id}, First Relevant Rank: {first_relevant_rank}, MRR: {current_mrr:.4f}")


if individual_mrr:
    mean_mrr = sum(individual_mrr.values()) / len(individual_mrr)
    print(f"\nMean Reciprocal Rank (MRR) for all queries: {mean_mrr:.4f}")
else:
    mean_mrr = 0.0
    print("No queries to evaluate.")


output_data = {
    'not_recalled': not_recalled,
    'mean_mrr': mean_mrr,
    'case_summary': case_summary,
    'individual_mrr': individual_mrr
}

output_dir = os.path.dirname(result_path)
output_file = os.path.join(output_dir, 'case_summary_with_mrr.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)
