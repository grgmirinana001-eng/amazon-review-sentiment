"""Independent arithmetic and edge-case checks; no API calls."""
import collections, json
from score_reviews import ROOT, metrics, truth
from emotions import derive

def verify():
    for mode in ['binary','balanced']:
        d=json.loads((ROOT/'results'/f'{mode}.json').read_text(encoding='utf-8')); rows=d['rows']; m=d['metrics']
        assert len(rows)==(100 if mode=='binary' else 150)
        assert len(set(r['source_line'] for r in rows))==len(rows)
        assert all(r['actual']==truth(r['rating'],mode=='binary') for r in rows)
        assert metrics(rows,m['classes'])==m
        assert all(r['correct']==(r['actual']==r['sentiment']) for r in rows)
        assert all(json.loads(r['raw_response']['choices'][0]['message']['content'])['sentiment']==r['sentiment'] for r in rows)
        if mode=='balanced':
            assert collections.Counter(r['actual'] for r in rows)==dict.fromkeys(m['classes'],50)
            assert d['emotion_metrics']['agree']==sum(r['emotion']==r['nrc_emotion'] for r in rows)
            assert all((r['nrc_emotion'] in r['nrc_tied']) if r['nrc_tied'] else r['nrc_emotion']=='none' for r in rows)
    assert truth(3,True)=='NEGATIVE' and truth(3)=='NEUTRAL'
    assert derive('','unmatched',{})['nrc_emotion']=='none'
    r=derive('happy','happy',{'happy':{'joy':1,'trust':1}})
    assert r['nrc_scores']['joy']==2 and r['nrc_emotion']=='joy' and r['nrc_tied']==['joy','trust']
    print('PASS: scoring, raw-output agreement, sample balance, labels, and NRC edge cases')

if __name__=='__main__': verify()
