import argparse
import json
from collections import defaultdict
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('--score_file', required=True)
parser.add_argument('--output_file',required=True)
parser.add_argument('--qrel_path',type=str,default='msmarco_passage/qrels.dev.tsv')
parser.add_argument('--eval_id_file',type=str,default=None)
parser.add_argument('--recall_path',type=str)
parser.add_argument('--tag',type=str,default=None)
args = parser.parse_args()
results={}
qrels={}
bm25_score={}
with open(args.recall_path) as f:
    for line in f:
        line = line.strip().split()
        qid = line[0]
        pid = line[2]
        rank = int(line[3])
        # if rank >= args.topk:
        #    continue
        score = float(line[4])
        if qid not in bm25_score:
            bm25_score[qid] = {}
        bm25_score[qid][pid] = score

with open(args.qrel_path) as f:
    for line in f:
        line = line.strip().split()
        qid = line[0]
        pid = line[2]
        if qid not in qrels:
            qrels[qid] = []
        qrels[qid].append(pid)
def write_result(results,fout):
    for qid in results:
        if args.tag != 'test' and qid not in qrels:
            no_judged += 1
            continue
        tem = {'qry':{'qid':qid},'psg':[]}
        res = results[qid]
        ar = 0
        sorted_res = sorted(res,key = lambda x:-x[-1])
        # print(sorted_res[:10])
        # exit()
        gold=False
        for i,ele in enumerate(sorted_res):
            pid = ele[0]
            label = 0
            if args.tag !='test' and pid in qrels[qid]:
                label = 1
                gold = True 
            tem['psg'].append({'pid':pid,'rank':i,'emb':ele[2],'score':ele[1],'label':label,'recall_score':bm25_score[qid][pid]})
            eval_lis[qid].append((pid,label))
        if not args.tag=='train' or gold is True:
            fout.write(json.dumps(tem)+'\n')
eval_lis=defaultdict(list)
with open(args.score_file) as f,open(args.output_file,'w') as fout:
    for line in tqdm(f):
        line = line.strip().split()
        qid = line[0]
        pid = line[2]
        # rank = int(line[3])
        score = float(line[4])
        emb = [float(x) for x in line[5:-1]]
        recall_score=bm25_score[qid][pid]
        if qid not in results:
            if len(results) > 7000:
                write_result(results,fout)
                del results
                results={}
            results[qid] = []
        results[qid].append((pid,score,emb,recall_score))
    if len(results) > 0:
        write_result(results,fout)
    
if args.tag !='train' and args.eval_id_file is not None:
    with open(args.eval_id_file,'w') as fout:
        json.dump(eval_lis,fout)
        


        
        
