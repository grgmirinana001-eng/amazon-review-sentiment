"""Download, sample, call an OpenAI-compatible API, and score saved predictions.
Python 3.11+; standard library only. API key is read from OPENAI_API_KEY.
"""
import argparse, collections, concurrent.futures, gzip, hashlib, json, os, pathlib, random, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz'
CLASSES = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'none']

def save(path, value):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def truth(rating, binary=False):
    return 'POSITIVE' if rating >= 4 else ('NEGATIVE' if binary or rating <= 2 else 'NEUTRAL')

def prepare(path, seed=6418, per_class=50):
    path = pathlib.Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True); urllib.request.urlretrieve(URL, path)
    rng = random.Random(seed); first = []; pools = {c: [] for c in CLASSES}; counts = collections.Counter(); stars = collections.Counter()
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            raw = json.loads(line)
            if raw['rating'] not in [1, 2, 3, 4, 5]: raise ValueError(f'Invalid rating at line {i+1}')
            r = {k: raw.get(k) for k in ['rating','title','text','verified_purchase','helpful_vote','timestamp','asin','parent_asin']}
            r['source_line'] = i + 1
            if not isinstance(r['title'], str) or not isinstance(r['text'], str): raise ValueError('Missing review text')
            if len(first) < 100: first.append(r)
            c = truth(r['rating']); counts[c] += 1; stars[str(int(r['rating']))] += 1
            if len(pools[c]) < per_class: pools[c].append(r)
            else:
                j = rng.randrange(counts[c])
                if j < per_class: pools[c][j] = r
    if any(len(pools[c]) != per_class for c in CLASSES): raise ValueError('Insufficient rows')
    balanced = [r for c in CLASSES for r in pools[c]]; rng.shuffle(balanced)
    save(ROOT/'results/samples.json', {'seed':seed,'first100':first,'balanced':balanced})
    save(ROOT/'results/dataset.json', {'source':URL,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(), 'total':sum(counts.values()), 'class_counts':dict(counts), 'star_counts':dict(stars), 'seed':seed, 'per_class':per_class})
    print(json.dumps({'total':sum(counts.values()),'classes':dict(counts),'first100':dict(collections.Counter(truth(r['rating'],True) for r in first))}))

def metrics(rows, classes):
    matrix = [[sum(r['actual']==a and r['sentiment']==p for r in rows) for p in classes] for a in classes]
    by = {}
    for i,c in enumerate(classes):
        tp=matrix[i][i]; support=sum(matrix[i]); predicted=sum(row[i] for row in matrix)
        precision=tp/predicted if predicted else 0; recall=tp/support if support else 0
        by[c]={'support':support,'predicted':predicted,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0}
    n=len(rows); correct=sum(matrix[i][i] for i in range(len(classes)))
    return {'n':n,'correct':correct,'accuracy':correct/n,'majority_baseline':max(sum(row) for row in matrix)/n,'macro_f1':sum(x['f1'] for x in by.values())/len(classes),'balanced_accuracy':sum(x['recall'] for x in by.values())/len(classes),'classes':classes,'matrix':matrix,'per_class':by,'mismatch_source_lines':[r['source_line'] for r in rows if not r['correct']]}

def call_model(title, text, prompt, base, model, key):
    payload={'model':model,'messages':[{'role':'system','content':prompt},{'role':'user','content':json.dumps({'title':title,'text':text},ensure_ascii=False)}], 'temperature':0, 'seed':6418, 'max_tokens':512, 'chat_template_kwargs':{'enable_thinking':False}}
    properties={'sentiment':{'type':'string','enum':CLASSES if 'NEUTRAL' in prompt else ['NEGATIVE','POSITIVE']}}
    if 'primary emotion' in prompt: properties['emotion']={'type':'string','enum':EMOTIONS}
    payload['response_format']={'type':'json_schema','json_schema':{'name':'classification','strict':True,'schema':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}}
    for attempt in range(3):
        try:
            req=urllib.request.Request(base.rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
            with urllib.request.urlopen(req,timeout=120) as resp: raw=json.load(resp)
            content=raw['choices'][0]['message']['content']; prediction=json.loads(content)
            if prediction.get('sentiment') not in CLASSES: raise ValueError('Invalid sentiment')
            if 'emotion' in prediction and prediction['emotion'] not in EMOTIONS: raise ValueError('Invalid emotion')
            return prediction, raw
        except Exception:
            if attempt==2: raise
            time.sleep(2**attempt)

def run(mode, base, model, workers):
    key=os.environ.get('OPENAI_API_KEY')
    if not key: raise SystemExit('Set OPENAI_API_KEY in your environment; never commit it.')
    binary=mode in ['binary','spot']; prompt=(ROOT/'prompts'/('binary.txt' if binary else 'three_class.txt')).read_text(encoding='utf-8')
    prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()
    if mode=='spot':
        samples=[{'source_line':-1,'title':'Fantastic','text':'Exactly what I wanted. Easy to use and I love it.','rating':5},{'source_line':-2,'title':'Useless','text':'It did not work. Very disappointed and I want a refund.','rating':1}]
    else: samples=json.loads((ROOT/'results/samples.json').read_text(encoding='utf-8'))['first100' if binary else 'balanced']
    cache_dir=ROOT/'results/cache'/mode; cache_dir.mkdir(parents=True,exist_ok=True)
    def classify(r):
        fingerprint=hashlib.sha256(json.dumps([r['title'],r['text'],prompt_hash,base,model,6418,0,'schema-v1'],ensure_ascii=False).encode()).hexdigest()
        cp=cache_dir/(fingerprint+'.json')
        if cp.exists(): saved=json.loads(cp.read_text(encoding='utf-8')); prediction=saved['prediction']; raw=saved['raw']
        else:
            prediction,raw=call_model(r['title'],r['text'],prompt,base,model,key)
            if binary and set(prediction)!={'sentiment'}: raise ValueError('Unexpected binary output')
            if binary and prediction['sentiment']=='NEUTRAL': raise ValueError('Neutral in binary output')
            if not binary and set(prediction)!={'sentiment','emotion'}: raise ValueError('Unexpected emotion output')
            save(cp,{'prediction':prediction,'raw':raw})
        actual=truth(r['rating'],binary)
        return {**r,**prediction,'actual':actual,'correct':prediction['sentiment']==actual,'raw_response':raw}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows=[]
        for row in pool.map(classify,samples):
            rows.append(row)
            if len(rows)%10==0: print(f'{mode}: {len(rows)}/{len(samples)}',flush=True)
    classes=['NEGATIVE','POSITIVE'] if binary else CLASSES
    result={'mode':mode,'model':model,'base_url':base,'temperature':0,'seed':6418,'response_format':'json_schema','prompt_sha256':prompt_hash,'metrics':metrics(rows,classes),'rows':rows}
    save(ROOT/'results'/f'{mode}.json',result); print(json.dumps(result['metrics']))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['prepare','spot','binary','balanced']); p.add_argument('--data',default=str(ROOT/'data/Gift_Cards.jsonl.gz')); p.add_argument('--base-url',default=os.getenv('OPENAI_BASE_URL','http://dobolyi.com:9001/v1')); p.add_argument('--model',default=os.getenv('OPENAI_MODEL','cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit')); p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if a.mode=='prepare': prepare(a.data)
    else: run(a.mode,a.base_url,a.model,a.workers)
