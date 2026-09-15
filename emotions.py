"""NRC exact-token emotion scoring; keeps all scores, ties and matched tokens."""
import argparse, collections, hashlib, html, json, pathlib, re, urllib.request
from score_reviews import ROOT, save

EMOTIONS=sorted(['anger','anticipation','disgust','fear','joy','sadness','surprise','trust'])
MIRROR='https://raw.githubusercontent.com/dinbav/LeXmo/master/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'

def load_lexicon(path):
    lex={}; entries=0
    for line in pathlib.Path(path).read_text(encoding='utf-8-sig').splitlines():
        bits=line.split('\t')
        if len(bits)!=3: continue
        word,emotion,value=bits
        if emotion not in EMOTIONS+['positive','negative'] or value not in ['0','1']: continue
        entries+=1; lex.setdefault(word,{})
        if emotion in EMOTIONS: lex[word][emotion]=int(value)
    if len(lex)!=14182 or entries!=141820: raise ValueError(f'Unexpected NRC file: {len(lex)} words, {entries} rows')
    return lex

def derive(title,text,lex):
    tokens=re.findall(r"[a-z]+(?:'[a-z]+)?",html.unescape(title+' '+text).lower())
    scores={e:0 for e in EMOTIONS}; matched=[]
    for token in tokens:
        vals=lex.get(token,{})
        if any(vals.values()): matched.append(token)
        for e,v in vals.items(): scores[e]+=v
    peak=max(scores.values()); tied=[e for e in EMOTIONS if scores[e]==peak] if peak else []
    return {'nrc_emotion':tied[0] if tied else 'none','nrc_scores':scores,'nrc_tied':tied,'nrc_words':matched}

def add(path,lexpath):
    lexpath=pathlib.Path(lexpath)
    if not lexpath.exists():
        lexpath.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve(MIRROR,lexpath)
    lex=load_lexicon(lexpath); run=json.loads(path.read_text(encoding='utf-8'))
    for r in run['rows']: r.update(derive(r['title'],r['text'],lex))
    rows=run['rows']; agree=sum(r['emotion']==r['nrc_emotion'] for r in rows)
    run['emotion_metrics']={'agree':agree,'agreement':agree/len(rows),'ties':sum(len(r['nrc_tied'])>1 for r in rows),'no_matches':sum(not r['nrc_tied'] for r in rows),'llm_counts':dict(collections.Counter(r['emotion'] for r in rows)),'nrc_counts':dict(collections.Counter(r['nrc_emotion'] for r in rows)),'pairs':dict(collections.Counter(r['emotion']+' → '+r['nrc_emotion'] for r in rows))}
    run['lexicon']={'name':'NRC Emotion Lexicon v0.92','homepage':'https://www.saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm','download_used':MIRROR,'sha256':hashlib.sha256(lexpath.read_bytes()).hexdigest(),'words':len(lex),'tie_rule':'alphabetical among maximum-scoring emotions; no matches = none'}
    save(path,run); print(json.dumps(run['emotion_metrics']))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=pathlib.Path,default=ROOT/'results/balanced.json'); p.add_argument('--lexicon',type=pathlib.Path,default=ROOT/'data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'); a=p.parse_args(); add(a.run,a.lexicon)
