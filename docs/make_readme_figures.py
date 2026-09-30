"""Regenerates the QUBO-v3 figures in docs/img from results/v3/campaign (ablation numbers: results/v3/qubo_v3)."""
import json, glob, collections, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
OUT=os.path.join(ROOT,'docs','img')
CAMP=os.path.join(ROOT,'results','v3','campaign')
BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'
INK,SEC,MUTED,GRID='#0b0b0b','#52514e','#8a8983','#e6e5e0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.edgecolor':MUTED,'axes.labelcolor':SEC,
 'xtick.color':SEC,'ytick.color':SEC,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
 'grid.color':GRID,'grid.linewidth':0.8,'axes.axisbelow':True,'figure.dpi':150,'savefig.bbox':'tight','legend.frameon':False})

# ---- data
runs=collections.defaultdict(dict); util={}
for arch in ['plain','priced']:
    for f in glob.glob(os.path.join(CAMP,arch,'*.json')):
        d=json.load(open(f)); sc=round(d['capacity_scale'],2)
        if d['utilization_percent']>=100: continue
        runs[(arch,sc,d['seed'])][d['method']]=(d['objective'],d['time_sec']); util[(sc,d['seed'])]=d['utilization_percent']
LB={}
for f in glob.glob(os.path.join(CAMP,'global','*.json')):
    for d in json.load(open(f)): LB[(round(d['capacity_scale'],2),d['seed'])]=(d['lagrangian_lower_bound'],d['feasible_objective'],d['time_sec'])

def gaps(arch,sc,m):
    return np.array([(v[m][0]-LB[(sc,s)][0])/LB[(sc,s)][0]*100 for (a,c,s),v in runs.items() if a==arch and c==sc and m in v])

# ---- Fig A: architecture effect
fig,ax=plt.subplots(figsize=(7,3.8))
scales=[1.0,0.33,0.25]
for arch,col,lab in [('plain',BLUE,'Per-camera window (K=5)'),('priced',ORANGE,'Per-camera window + capacity prices')]:
    xs=[np.mean([util[k] for k in util if k[0]==sc and util[k]<100]) for sc in scales]
    g=[gaps(arch,sc,'sqa_v3') for sc in scales]
    m=[x.mean() for x in g]; ci=[1.96*x.std(ddof=1)/np.sqrt(len(x)) for x in g]
    ax.errorbar(xs,m,yerr=ci,color=col,lw=2,marker='o',ms=6,capsize=3,label=lab)
    ax.annotate(f'{m[-1]:.1f}%',(xs[-1],m[-1]),xytext=(8,0),textcoords='offset points',va='center',color=SEC)
ax.plot([23.77,95.07],[96.4,88.2],color=MUTED,lw=2,ls='--',marker='s',ms=6,label='Shared 80×20 window (seed 42 only)')
ax.set_xlabel('Server utilisation (%)'); ax.set_ylabel('Gap to global lower bound (%)')
ax.set_xlim(15,105); ax.set_ylim(-3,105)
ax.set_title('SQA-v3: effect of the candidate window and capacity prices',loc='left',color=INK,fontsize=11)
ax.legend(loc='center left',bbox_to_anchor=(0.0,0.62))
fig.savefig(f'{OUT}/v3_architecture_effect.png'); plt.close(fig)

# ---- Fig B: ablation ladder (12 hard batches, seed 42)
rows=[('Base (one-hot)',0.715,1.433),('+ exact reduction (R)',0.162,0.167),('+ domain wall (DW)',0.379,0.260),
      ('R + DW',0.081,0.034),('R + DW + adaptive prices',0.066,0.024),('… + tuned SQA (β=20)',0.035,None)]
fig,ax=plt.subplots(figsize=(7,3.6))
for i,(lab,sqa,sa) in enumerate(rows[::-1]):
    if sa: ax.plot([sa,sqa],[i,i],color=GRID,lw=2,zorder=1); ax.scatter(sa,i,s=50,facecolor='white',edgecolor=BLUE,lw=2,zorder=3,label='SA' if i==1 else None)
    ax.scatter(sqa,i,s=50,color=ORANGE,zorder=3,edgecolor='white',lw=1,label='SQA' if i==0 else None)
for v,lab,ha in [(0.676,'greedy','center'),(0.016,'regret','left'),(0.010,'regret+LS','right')]:
    ax.axvline(v,color=MUTED,ls=':',lw=1.2); ax.text(v*(1.06 if ha=='left' else 0.94 if ha=='right' else 1),len(rows)-0.35,lab,ha=ha,color=MUTED,fontsize=8.5)
ax.set_xscale('log'); ax.set_xlim(0.006,2.2); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.set_ylim(-0.6,len(rows)-0.1); ax.grid(axis='y',visible=False)
ax.set_xticks([0.01,0.03,0.1,0.3,1]); ax.set_xticklabels(['0.01%','0.03%','0.1%','0.3%','1%'])
ax.set_xlabel('Gap to the exact batch optimum (log scale)')
ax.set_title('QUBO-v3 ablation on 12 hard batches (95% utilisation, per-camera window)',loc='left',color=INK,fontsize=11)
h,l=ax.get_legend_handles_labels(); ax.legend(h[::-1],l[::-1],loc='lower right')
fig.savefig(f'{OUT}/v3_ablation.png'); plt.close(fig)

# ---- Fig C: paired differences at ~97%
fig,ax=plt.subplots(figsize=(7,3.6)); rng=np.random.default_rng(3)
groups=['greedy','regret','exact','sa_v3']; names=['vs greedy','vs regret','vs exact batch MILP','vs SA-v3']
for gi,m in enumerate(groups):
    for ai,(arch,col) in enumerate([('plain',BLUE),('priced',ORANGE)]):
        d=np.array([v['sqa_v3'][0]-v[m][0] for (a,c,s),v in runs.items() if a==arch and c==0.25])
        x=gi+(-0.17 if ai==0 else 0.17); dc=np.clip(d,-260,260)
        ax.scatter(x+rng.uniform(-0.07,0.07,len(d)),dc,s=26,color=col,alpha=0.85,edgecolor='white',lw=0.6,
                   label=('without prices' if ai==0 else 'with prices') if gi==0 else None,zorder=3)
        ax.plot([x-0.1,x+0.1],[d.mean()]*2,color=INK,lw=2,zorder=4)
        out=sorted(d[np.abs(d)>260])
        if out: ax.annotate(', '.join(f'{v:.0f}' for v in out)+' ↓',(x,-266 if ai==0 else -283),ha='center',va='top',fontsize=7,color=col)
ax.axhline(0,color=SEC,lw=1)
ax.set_xticks(range(4)); ax.set_xticklabels(names); ax.set_ylim(-300,300); ax.grid(axis='x',visible=False)
ax.set_ylabel('SQA-v3 objective minus method\n(< 0: SQA-v3 better)')
ax.set_title('Per-seed paired differences at ~97% utilisation (9 seeds; bar = mean)',loc='left',color=INK,fontsize=11)
ax.legend(loc='upper left',ncol=2)
fig.savefig(f'{OUT}/v3_paired_differences.png'); plt.close(fig)

# ---- Fig D: quality vs time (priced, ~97%)
fig,ax=plt.subplots(figsize=(7,3.6))
lab={'greedy':'greedy','regret':'regret','exact':'exact batch MILP','sa_v3':'SA-v3','sqa_v3':'SQA-v3'}
col={'greedy':MUTED,'regret':MUTED,'exact':SEC,'sa_v3':BLUE,'sqa_v3':ORANGE}
off={'greedy':(8,4),'regret':(8,6),'exact':(-6,-14),'sa_v3':(8,-12),'sqa_v3':(-10,8)}
for m in lab:
    t=np.array([v[m][1] for (a,c,s),v in runs.items() if a=='priced' and c==0.25]); g=gaps('priced',0.25,m)
    ax.scatter(t.mean(),g.mean(),s=60,color=col[m],edgecolor='white',lw=1,zorder=3)
    ax.annotate(lab[m],(t.mean(),g.mean()),xytext=off[m],textcoords='offset points',color=SEC,fontsize=9)
gl=[(LB[k][1]-LB[k][0])/LB[k][0]*100 for k in LB if k[0]==0.25 and util.get(k,101)<100]; gt=[LB[k][2] for k in LB if k[0]==0.25 and util.get(k,101)<100]
ax.scatter(np.mean(gt),np.mean(gl),s=60,color=AQUA,edgecolor='white',lw=1,zorder=3)
ax.annotate('global solution (offline reference)',(np.mean(gt),np.mean(gl)),xytext=(8,4),textcoords='offset points',color=SEC,fontsize=9)
ax.set_xscale('log'); ax.set_xlim(0.3,600); ax.set_ylim(0,24)
import matplotlib.ticker as mt
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v,p: f'{v:g}'))
ax.set_xlabel('Wall-clock time per instance, s (log scale, CPU)'); ax.set_ylabel('Gap to global lower bound (%)')
ax.set_title('Quality vs time at ~97% utilisation, with capacity prices (means over 9 seeds)',loc='left',color=INK,fontsize=11)
fig.savefig(f'{OUT}/v3_quality_vs_time.png'); plt.close(fig)
print('ok')
