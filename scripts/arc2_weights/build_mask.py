import json, glob, hashlib, sys
A='./.claude/worktrees/arc2'
HERE=__import__('os').path.dirname(__file__)
u=json.load(open(HERE+'/masks/usage-c240-weight-count.json'))  # usage-c240.json's weight/count rows
W=u['weight']; C=u['count']
k256=json.load(open(sorted(glob.glob(A+'/dist/*/configs/expert-prune/prune-arc2-k256-v1.json'))[0]))
E,L,G,TOTAL=512,48,32,256*48
order={}; share={}
for l in range(L):
    tot=sum(W[l])
    o=sorted(range(E),key=lambda e:(W[l][e],C[l][e]))  # least mass first
    order[l]=o; share[l]=[W[l][e]/tot for e in o]
# check k256 = 256 least by weight
same=sum(set(order[l][:256])=={int(e) for e in k256['pruned'][str(l)]} for l in range(L)); print('k256 reproduced layers',same)
def kept_for(tau):
    ks={}
    for l in range(L):
        cum=0; npr=0
        for s in share[l]:
            if cum+s>tau: break
            cum+=s; npr+=1
        kept=E-npr; kept=-(-kept//G)*G; ks[l]=min(E,max(G,kept))
    return ks
lo,hi=0.0,0.5
for _ in range(60):
    m=(lo+hi)/2
    if sum(kept_for(m).values())>TOTAL: lo=m
    else: hi=m
ks=kept_for(hi); tau=hi
print('tau',tau,'total',sum(ks.values()),'min',min(ks.values()),'max',max(ks.values()))
drop={l:sum(share[l][:E-ks[l]]) for l in range(L)}
d256={l:sum(share[l][:256]) for l in range(L)}
print('dropped new mean %.4f max %.4f min %.4f | k256 mean %.4f max %.4f min %.4f'%(sum(drop.values())/L,max(drop.values()),min(drop.values()),sum(d256.values())/L,max(d256.values()),min(d256.values())))
print('kept',[ks[l] for l in range(L)])
pruned={str(l):sorted(order[l][:E-ks[l]]) for l in range(L)}
newexp=sum(len(set(range(E))-set(pruned[str(l)]) - (set(range(E))-{int(e) for e in k256['pruned'][str(l)]})) for l in range(L))
print('kept experts not in k256:',newexp)
m={'format':'arc-expert-prune-v1','experts':E,'fraction':round(sum(len(v) for v in pruned.values())/(E*L),4),
 'pruned_per_layer':'per-layer','pruned_counts':{k:len(v) for k,v in pruned.items()},'pruned_total':sum(len(v) for v in pruned.values()),
 'pruned':pruned,'rank_by':'weight',
 'allocation':'layer-aware: equal dropped c240 routing mass per layer (water-filling, tau %.4f), kept counts rounded up to multiples of 32, mean 256 kept; least gate-weight mass first, routing count breaks ties'%tau,
 'dropped_mass':{'mean':round(sum(drop.values())/L,4),'max':round(max(drop.values()),4),'min':round(min(drop.values()),4)},
 'source_run':k256['source_run']}
out=sys.argv[1]
raw=json.dumps(m,indent=1,sort_keys=True).encode(); open(out,'wb').write(raw); print(out,hashlib.sha256(raw).hexdigest()[:8])
