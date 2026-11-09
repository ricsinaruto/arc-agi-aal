import os
os.chdir(os.environ.get('L4PP_DATA','.'))  # folder with limits/ and sftspeed/ (router usage, mask)
# PP step model. Anchored on l4prof decode-shallow (79 ms, 32 streams, MTP 2, 96 token rows, TP4+EP) and prefill-56k.
import json, numpy as np, sys
d=json.load(open('limits/r15/expert-usage.json')); cnt=np.array(d['usage']['count'],float)
m=json.load(open('sftspeed/reap/prune-p75-mass-v1.json'))
pr=np.zeros((48,512),bool)
for l,e in m['pruned'].items(): pr[int(l),e]=True
K=10
def probs(c):
    p=c/c.sum(); pi=K*p
    for _ in range(50):
        o=pi>1
        if not o.any(): break
        i=~o; pi=np.where(o,1,pi); pi[i]=pi[i]/pi[i].sum()*(K-o.sum())
    return pi
PK=[probs(cnt[l][~pr[l]]) for l in range(48)]; PU=[probs(cnt[l]) for l in range(48)]
K0=int(os.environ.get('L4PP_K0','10'))  # OEA: the experts loaded are the union of each token's top k0
K=K0; PK=[probs(cnt[l][~pr[l]]) for l in range(48)]; K=10
F=.8  # effective-row factor fitted to the 9 Oct one-layer probe (133/179/204 at 24/40/56 rows)
tau =lambda r: float(np.mean([(1-(1-p)**(r*F)).sum() for p in PK]))
tauu=lambda r: float(np.mean([(1-(1-p)**(r*F)).sum() for p in PU]))
US=8.7e-3 # ms per expert touched
a4=lambda r: float(np.interp(r,[1,16,32,40,48,64,96,1600],[23,59,106,119,139,154,240,2532]))*1e-3  # Kaggle in-graph probe v2/v4, step profile
def a2(r,hi=False): return (12e-3+(a4(r)-23e-3)*.67)*(1.4 if hi else 1)  # assumed: half the fixed cost, 2/3 of the ring bytes
def comp(N,s,p,hi=False):
    t=4//p; R=N*(1+s); r=R/p; n=N/p
    e=(48/p)*tau(r)/t*US
    ar=0 if t==1 else (103 if s else 96)/p*(a4(r) if t==4 else a2(r,hi))+ (1.0/p)
    i8=max(3.3,8.4*r/96)
    hg=(201/p)*max(.028,.038*r/96)
    ho=2.1*(r/96)*(4/p)/4 if False else 2.1/p*(r/96)*p/4*4/4
    ho=2.1*(r/96)/(1 if p==1 else 1)*(1/1)*(1 if p==1 else 1/1)*(1/p)*p/p if False else 2.1*(R/96)/p/p
    router=1.0/p
    rest=6.75/p+6.75*(r/96) - (0 if s else 3.0/p)
    if not s: rest=max(rest,1)
    samp=3.6*(n/32)
    dr=0 if not s else 2*tauu(n)/t*.0109 + 3.0*(.5/p*1 + .5*(N/32)/p/p*p/1 if False else (.5+.5*(N/32)/p))
    hop=0 if p==1 else .03+r*20480/5e9*1e3
    base=e+ar+i8+hg+ho+router+rest
    last=base+samp+dr
    return dict(p=p,t=t,e=e,ar=ar,i8=i8,hg=hg+ho,rest=rest,samp=samp,dr=dr,hop=hop,base=base,last=last)
def rate(N,s,p,hi=False):
    c=comp(N,s,p,hi); M=2.54 if s else 1
    if p==1: step=c['last']; return N/step*M*1e3, step, step, c
    bal=(c['base']*p+c['samp']+c['dr'])/p+c['hop']; un=c['last']+c['hop']
    return (N/p)/bal*M*1e3,(N/p)/un*M*1e3,bal*p,c
print('tau pruned', {r:round(tau(r)) for r in (8,14,16,24,27,32,48,54,96)})
for N,s in ((32,2),(45,2),(60,2),(32,0),(35,0),(54,0),(60,0),(80,0)):
    tp=rate(N,s,1)
    print(f'N={N} MTP{s}: TP4 step {tp[1]:.1f} ms {tp[0]:.0f} tok/s')
    for p in (4,2):
        for hi in ((False,) if p==4 else (False,True)):
            b,u,st,c=rate(N,s,p,hi)
            x=f"PP{p}" if p==4 else f"PP2TP2{' (a2 x1.4)' if hi else ''}"
            print(f'   {x:18s} balanced {b:5.0f} ({b/tp[0]:.2f}x) unbalanced {u:5.0f} ({u/tp[0]:.2f}x) cycle {st/p:.1f} ms; token gap {st:.0f} ms; exp {c["e"]:.1f} ar {c["ar"]:.1f} i8 {c["i8"]:.1f} hc {c["hg"]:.1f} rest {c["rest"]:.1f} samp+dr {c["samp"]+c["dr"]:.1f} hop {c["hop"]:.2f}')
# prefill 56K, 1,383 rows per step
T=dict(ar=253,g=118,gap=88,hc=63,e=37,rest=76); tot=sum(T.values()); rows=2178*.635
hop=.03+rows*20480/5e9*1e3
pp4=T['g']+T['gap']+T['hc']/4+T['e']+T['rest']+hop
pp2=T['g']+T['gap']+T['hc']/2+T['e']+T['rest']+hop+52*(2.46*.67)
print('prefill TP4',tot,'rows',round(rows),'hop',round(hop,1),'PP4',round(pp4),round(tot/pp4,2),'PP2TP2',round(pp2),round(tot/pp2,2))
