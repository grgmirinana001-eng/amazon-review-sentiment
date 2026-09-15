"""Generate an offline dashboard from saved evaluation JSON. No web dependencies."""
import json
from score_reviews import ROOT

def generate():
    runs={}
    for name in ['binary','balanced']:
        p=ROOT/'results'/f'{name}.json'
        if p.exists():
            run=json.loads(p.read_text(encoding='utf-8'))
            for r in run['rows']: r.pop('raw_response',None)
            runs[name]=run
    data={'runs':runs,'dataset':json.loads((ROOT/'results/dataset.json').read_text())}
    template=(ROOT/'dashboard_template.html').read_text(encoding='utf-8')
    (ROOT/'dashboard.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')),encoding='utf-8')
    print('Generated dashboard.html')

if __name__=='__main__': generate()
