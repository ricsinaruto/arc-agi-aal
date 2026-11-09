"""Resolve a docs/modal-queue.json merge conflict: 3-way merge of rows by run name (theirs wins on both-changed rows). Run from the repo root while MERGE_HEAD exists, then git add + commit."""
import json,subprocess,sys
def load(ref): return json.loads(subprocess.check_output(['git','show',f'{ref}:docs/modal-queue.json']))
B=subprocess.check_output(['git','merge-base','HEAD','MERGE_HEAD']).decode().strip()
base,ours,theirs=load(B),load('HEAD'),load('MERGE_HEAD')
def key(r): return r.get('run') if isinstance(r,dict) else json.dumps(r,sort_keys=True)
def mlist(b,o,t):
    bk={key(r):r for r in b}; ok={key(r):r for r in o}; tk={key(r):r for r in t}
    out=list(t)
    for k,r in ok.items():
        if k in tk: continue
        if k in bk: continue   # removed upstream
        out.append(r)          # added locally
    # rows removed locally but kept upstream unchanged -> drop
    out=[r for r in out if not (key(r) in bk and key(r) not in ok and tk.get(key(r))==bk[key(r)])]
    return out
def mdict(b,o,t):
    out=dict(t)
    for k,v in o.items():
        if k not in t:
            if k not in b: out[k]=v
        elif isinstance(v,list) and isinstance(t[k],list): out[k]=mlist(b.get(k,[]) if isinstance(b.get(k),list) else [],v,t[k])
        elif isinstance(v,dict) and isinstance(t[k],dict): out[k]=mdict(b.get(k,{}) if isinstance(b.get(k),dict) else {},v,t[k])
        elif t[k]==b.get(k) and v!=b.get(k): out[k]=v
    for k in list(out):
        if k in b and k not in o and t.get(k)==b[k]: del out[k]
    return out
m=mdict(base,ours,theirs)
open('docs/modal-queue.json','w').write(json.dumps(m,indent=1,ensure_ascii=False)+'\n')
print([ (k,len(v)) for k,v in m.items() if isinstance(v,list)])
print([r['run'] for r in m.get('live',[])])
