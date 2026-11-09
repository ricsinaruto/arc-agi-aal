import json, numpy as np
d=json.load(open('limits/r15/expert-usage.json'))
ug=d['usage']; print([k for k in ug.keys()])
cnt=np.array(ug['count'],float) if 'count' in ug else None
print(cnt.shape, 'rows tokens', cnt.sum(1)[:3])
m=json.load(open('sftspeed/reap/prune-p75-mass-v1.json'))
pr=np.zeros((48,512),bool)
for l,e in m['pruned'].items(): pr[int(l),e]=True
K=10
def probs(c):
    p=c/c.sum()
    # per-token inclusion prob pi_i = min(1,K*p) with redistribution
    pi=K*p
    for _ in range(50):
        over=pi>1
        if not over.any(): break
        rest=K-over.sum()
        pi=np.where(over,1,pi); 
        idx=~over
        pi[idx]=pi[idx]/pi[idx].sum()*rest if pi[idx].sum()>0 else 0
    return pi
def touched(pi,r): return (1-(1-pi)**r).sum()
rows=[1,8,14,24,32,40,42,48,54,56,96,160]
un=np.mean([[touched(probs(cnt[l]),r) for r in rows] for l in range(48)],0)
pm=np.mean([[touched(probs(cnt[l][~pr[l]]),r) for r in rows] for l in range(48)],0)
print('rows',rows)
print('unpruned model',np.round(un,1))
print('measured probe us/8.7: 1:10.5 8:63 24:133 40:179 56:204 (one layer)')
print('pruned128 model',np.round(pm,1))
# correlated scaling: effective rows factor so unpruned model matches probe
for f in (1,.8,.7,.6,.5):
    un2=[np.mean([touched(probs(cnt[l]),r*f) for l in range(48)]) for r in (8,24,40,56)]
    print('f',f,np.round(un2,0))
