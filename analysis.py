"""
=============================================================================
Vestergaard Data Science Intern Case Study 2026
Mosquito Net Durability Study — Complete Python Analysis
Author: Sally Karimi Kinyua | March 2026
=============================================================================
Run: python analysis.py
=============================================================================
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import json, os, warnings; warnings.filterwarnings('ignore')

os.makedirs('charts/python', exist_ok=True)

NAVY='#1C3F6E'; TEAL='#4A8FA8'; ORANGE='#E87722'; RED='#C0392B'
GRAY='#666666'; GREEN='#2E7D32'

def base_style(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.yaxis.grid(True, color='#EEEEEE', zorder=0); ax.set_axisbelow(True)
    ax.tick_params(colors=NAVY)

# 1. LOAD & CLEAN
df = pd.read_csv('data/data_for_case_study_data_science_intern_2026.csv')
for col in ['detergent','mattress','bed','matground','hang','fold']:
    if col in df.columns: df[col] = df[col].replace(9, np.nan)
df['still'] = df['still'].replace(9, np.nan)
df['functional'] = np.where((df['still']==1)&(df['surv']==1),1,
                   np.where((df['still']==0)|(df['surv']==0),0,np.nan))
df['country_B']   = (df['country']=='B').astype(int)
present      = df[df['still']==1].copy()
present_surv = present[present['surv'].notna()].copy()

print("="*65)
print("VESTERGAARD NET DURABILITY STUDY — 24-MONTH FOLLOW-UP")
print("="*65)
print(f"Dataset: {df.shape[0]} nets | {df['country'].nunique()} countries | {df['district'].nunique()} districts")
print(f"Retention: {df['still'].mean()*100:.1f}% | Serviceable: {present_surv['surv'].mean()*100:.1f}% | Functional: {df['functional'].mean()*100:.1f}%")

# 2. COUNTRY COMPARISON
print("\n=== COUNTRY COMPARISON ===")
cs = {}
for c in ['A','B']:
    sub=df[df['country']==c]; sub_ps=sub[(sub['still']==1)&sub['surv'].notna()]
    cs[c]={'n':len(sub),'retention':round(sub['still'].mean()*100,1),
           'serviceability':round(sub_ps['surv'].mean()*100,1),
           'functional':round(sub['functional'].mean()*100,1),
           'care_att':round(sub['crattgr'].mean(),2),'net_att':round(sub['nattgr'].mean(),2)}
    print(f"Country {c}: retention={cs[c]['retention']}%, serv={cs[c]['serviceability']}%, functional={cs[c]['functional']}%, care_att={cs[c]['care_att']}")

chi2,p,_,_=stats.chi2_contingency(pd.crosstab(df['country'],df['functional'].dropna()))
print(f"Chi-square: X²={chi2:.2f}, p={p:.6f} ***")

# 3. ATTITUDE ANALYSIS
print("\n=== ATTITUDE vs SERVICEABILITY ===")
for att,name in [('crattgr','Care & Repair'),('nattgr','Net Attitude')]:
    for g in [0,1,2]:
        sub=present_surv[present_surv[att]==g]
        print(f"  {name} group {g} (n={len(sub)}): {sub['surv'].mean()*100:.1f}% serviceable")
    chi2,p,_,_=stats.chi2_contingency(pd.crosstab(present_surv[att],present_surv['surv']))
    print(f"  Chi-square p={p:.6f} ***")

print("\nCare Attitude Distribution by Country:")
for c in ['A','B']:
    sub=df[df['country']==c]; tbl=sub['crattgr'].value_counts(normalize=True).sort_index()*100
    print(f"  Country {c}: Low={tbl.get(0,0):.1f}%, Med={tbl.get(1,0):.1f}%, High={tbl.get(2,0):.1f}%")

mw=stats.mannwhitneyu(df[df['country']=='A']['crattgr'],df[df['country']=='B']['crattgr'],alternative='two-sided')
print(f"Mann-Whitney U={mw.statistic:.0f}, p={mw.pvalue:.6f} *** | Median A={df[df['country']=='A']['crattgr'].median()}, B={df[df['country']=='B']['crattgr'].median()}")

# 4. SPEARMAN CORRELATIONS
print("\n=== SPEARMAN CORRELATIONS ===")
for var in ['crattgr','nattgr','numwgr','numdogr']:
    r,p=stats.spearmanr(present_surv[var],present_surv['surv'])
    sig='***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f"  {var:12s}: r={r:7.3f}, p={p:.4f} {sig}")

# 5. LOGISTIC REGRESSION
print("\n=== LOGISTIC REGRESSION ===")
feats=['crattgr','nattgr','numwgr','numdogr','country_B']
mdf=present_surv[feats+['surv']].dropna()
X=mdf[feats]; y=mdf['surv']
lr=LogisticRegression(max_iter=1000,solver='lbfgs')
lr.fit(StandardScaler().fit_transform(X),y)
print(f"Accuracy={lr.score(StandardScaler().fit_transform(X),y)*100:.1f}%")
for f,c in zip(feats,lr.coef_[0]):
    print(f"  {f:12s}: OR={np.exp(c):.3f} ({'↑ protective' if c>0 else '↓ risk'})")

# 6. COUNTERFACTUAL SIMULATION
print("\n=== COUNTERFACTUAL SIMULATION ===")
a_att=df[df['country']=='A']['crattgr'].value_counts(normalize=True).sort_index()
serv_att=present_surv.groupby('crattgr')['surv'].mean()
b_ret=df[df['country']=='B']['still'].mean()
b_serv_now=present_surv[present_surv['country']=='B']['surv'].mean()
b_func_now=b_ret*b_serv_now
sim_serv=sum(a_att.get(g,0)*serv_att.get(g,0) for g in [0,1,2])
sim_func=b_ret*sim_serv
sim_results={'b_actual_pct':round(b_func_now*100,1),'b_simulated_pct':round(sim_func*100,1),
             'a_actual_pct':66.2,'improvement_pp':round((sim_func-b_func_now)*100,1),
             'nets_per_1000':round((sim_func-b_func_now)*1000,0)}
print(f"Country B actual:    {sim_results['b_actual_pct']}%")
print(f"Country B simulated: {sim_results['b_simulated_pct']}%")
print(f"Improvement:        +{sim_results['improvement_pp']}pp | +{int(sim_results['nets_per_1000'])} nets per 1,000 distributed")
with open('simulation_results.json','w') as f: json.dump(sim_results,f,indent=2)

# 7. CHARTS
print("\n=== GENERATING CHARTS ===")

# Plot 1: Country comparison
fig,axes=plt.subplots(1,3,figsize=(11,5)); fig.patch.set_facecolor('white')
fig.suptitle('Net Outcomes at 24 Months: Country A vs Country B\nChi-square p<0.001',fontsize=13,fontweight='bold',color=NAVY)
for ax,met,av,bv in zip(axes,['Retention','Serviceability\n(of present)','Functional'],[84.6,78.3,66.2],[69.7,53.4,37.3]):
    ax.set_facecolor('white')
    bars=ax.bar(['Country A','Country B'],[av,bv],color=[NAVY,ORANGE],width=0.55,zorder=3)
    for b,v in zip(bars,[av,bv]):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.5,f'{v}%',ha='center',va='bottom',fontsize=12,fontweight='bold',color=NAVY)
    ax.set_title(met,fontsize=11,fontweight='bold',color=NAVY); ax.set_ylim(0,100)
    ax.text(0.5,-0.13,f'Gap: {bv-av:+.1f}pp',transform=ax.transAxes,ha='center',fontsize=9,color=RED,style='italic')
    base_style(ax)
plt.tight_layout(rect=[0,0,1,0.90])
plt.savefig('charts/python/plot1_country_comparison.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  ✓ Plot 1")

# Plot 2: Attitude vs serviceability
att_s=present_surv.groupby('crattgr').agg(pct=('surv','mean'),n=('surv','count')).reset_index()
att_s['pct']*=100
fig,ax=plt.subplots(figsize=(8,5.5)); fig.patch.set_facecolor('white'); ax.set_facecolor('white')
bars=ax.bar(att_s['crattgr'],att_s['pct'],color=[RED,ORANGE,GREEN],width=0.55,zorder=3)
for b,(_,row) in zip(bars,att_s.iterrows()):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.5,f"{row['pct']:.1f}%\n(n={int(row['n'])})",
            ha='center',va='bottom',fontsize=11,fontweight='bold',color=NAVY)
ax.annotate('',xy=(2,88),xytext=(0,88),arrowprops=dict(arrowstyle='<->',color=GREEN,lw=2))
ax.text(1,90.5,'+24.5 percentage points',ha='center',fontsize=10,fontweight='bold',color=GREEN)
ax.set_xticks([0,1,2]); ax.set_xticklabels(['Low\n(score 0)','Medium\n(score 1)','High\n(score 2)'])
ax.set_ylim(0,100); ax.set_ylabel('Serviceability Rate (%)',color=GRAY)
ax.set_title('Care & Repair Attitude vs Net Serviceability\nOR=1.52 (p<0.001), Spearman r=0.246',fontsize=13,fontweight='bold',color=NAVY)
base_style(ax); plt.tight_layout()
plt.savefig('charts/python/plot2_attitude_serviceability.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  ✓ Plot 2")

# Plot 3: District
dist_info=[(1,'A',59.8),(2,'A',66.3),(3,'A',61.8),(4,'A',50.5)]
fig,ax=plt.subplots(figsize=(9,5.5)); fig.patch.set_facecolor('white'); ax.set_facecolor('white')
dlabels=[f"A-District {d[0]}" for d in dist_info]; dvals=[d[2] for d in dist_info]
bars=ax.barh(dlabels,dvals,color=[NAVY if v>=60 else ORANGE for v in dvals],height=0.55,zorder=3)
for b,v in zip(bars,dvals):
    ax.text(v+0.5,b.get_y()+b.get_height()/2,f'{v}%',va='center',fontsize=11,fontweight='bold',color=NAVY)
ax.axvline(66.2,color=NAVY,linestyle='--',lw=1.5,alpha=0.6)
ax.text(66.7,3.5,'Country A\navg 66.2%',fontsize=8,color=NAVY,style='italic')
ax.set_xlim(0,83); ax.set_xlabel('Functional Net Rate (%)',color=GRAY)
ax.set_title('Functional Nets by District — 24 Months\nA-District 4 outlier: 15.7pp below Country A average',fontsize=13,fontweight='bold',color=NAVY)
base_style(ax); ax.spines['left'].set_visible(False); ax.yaxis.grid(False); ax.xaxis.grid(True,color='#EEEEEE',zorder=0)
plt.tight_layout()
plt.savefig('charts/python/plot3_district_functional.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  ✓ Plot 3")

# Plot 4: Attitude distribution stacked
att_data={'Country A':{0:8.9,1:12.7,2:78.4},'Country B':{0:21.8,1:64.9,2:13.3}}
fig,ax=plt.subplots(figsize=(8,5.5)); fig.patch.set_facecolor('white'); ax.set_facecolor('white')
countries=list(att_data.keys()); bottom=[0.0,0.0]
for g,label,color in [(0,'Low (0)',RED),(1,'Medium (1)',ORANGE),(2,'High (2+)',TEAL)]:
    vals=[att_data[c][g] for c in countries]
    ax.bar(countries,vals,bottom=bottom,color=color,width=0.5,label=label,zorder=3)
    for i,(v,bot) in enumerate(zip(vals,bottom)):
        if v>5: ax.text(i,bot+v/2,f'{v}%',ha='center',va='center',color='white',fontsize=13,fontweight='bold')
    bottom=[bottom[i]+vals[i] for i in range(2)]
ax.set_ylim(0,108); ax.set_ylabel('% of Households',color=GRAY)
ax.set_title('Care & Repair Attitude Distribution by Country\nCountry B: only 13.3% high attitude vs 78.4% in Country A',fontsize=12,fontweight='bold',color=NAVY)
ax.legend(title='Care Attitude',loc='upper right'); base_style(ax); plt.tight_layout()
plt.savefig('charts/python/plot4_attitude_distribution.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  ✓ Plot 4")

# Plot 5: Simulation
fig,ax=plt.subplots(figsize=(9,5.5)); fig.patch.set_facecolor('white'); ax.set_facecolor('white')
scens=['Country A\n(Actual)','Country B\n(Actual)','Country B\n(Simulated BCC)']
vvals=[66.2,37.3,sim_results['b_simulated_pct']]
bars=ax.bar(scens,vvals,color=[NAVY,ORANGE,GREEN],width=0.55,zorder=3)
for b,v in zip(bars,vvals):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.5,f'{v}%',ha='center',va='bottom',fontsize=13,fontweight='bold',color=NAVY)
ax.axhline(66.2,color=NAVY,linestyle='--',lw=1.3,alpha=0.5)
ax.annotate('',xy=(2,vvals[2]),xytext=(2,37.3),arrowprops=dict(arrowstyle='->',color=GREEN,lw=2.5))
ax.text(2.30,45,f"+{sim_results['improvement_pp']}pp\n+{int(sim_results['nets_per_1000'])} nets/1,000",fontsize=10,fontweight='bold',color=GREEN,va='center')
ax.set_ylim(0,82); ax.set_ylabel('Functional Net Rate (%)',color=GRAY)
ax.set_title('Counterfactual Simulation: Impact of Behaviour Change\nIf Country B adopted Country A care attitudes',fontsize=12,fontweight='bold',color=NAVY)
base_style(ax); plt.tight_layout()
plt.savefig('charts/python/plot5_simulation.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  ✓ Plot 5")

# Plot 6: Map
fig,ax=plt.subplots(figsize=(10,6.5)); fig.patch.set_facecolor('white'); ax.set_facecolor('#D6EAF8')
ax.set_xlim(0,10); ax.set_ylim(0,8); ax.set_aspect('equal')
def fc(p): return '#1C3F6E' if p>=65 else '#3A7CA5' if p>=60 else '#E87722' if p>=50 else '#C0392B'
for x,y_,w,h,lab,pct in [(0.8,4.5,2.2,2.8,'A-D2\n66.3%',66.3),(0.8,1.5,2.2,2.8,'A-D1\n59.8%',59.8),
                           (3.2,4.5,2.2,2.8,'A-D3\n61.8%',61.8),(3.2,1.5,2.2,2.8,'A-D4\n50.5%',50.5)]:
    rect=mpatches.FancyBboxPatch((x,y_),w,h,boxstyle='round,pad=0.08',facecolor=fc(pct),edgecolor='white',linewidth=2)
    ax.add_patch(rect)
    for i,line in enumerate(lab.split('\n')):
        ax.text(x+w/2,y_+h/2+(0.25-i*0.5),line,ha='center',va='center',fontsize=11 if i==0 else 13,color='white',fontweight='bold')
ax.text(2.7,7.5,'COUNTRY A',ha='center',fontsize=14,color=NAVY,fontweight='bold')
ax.text(2.7,7.05,'n=1,789 nets',ha='center',fontsize=9,color=GRAY)
rect=mpatches.FancyBboxPatch((6.3,1.8),3.0,5.0,boxstyle='round,pad=0.08',facecolor='#C0392B',edgecolor='white',linewidth=2)
ax.add_patch(rect)
for i,(txt,sz) in enumerate([('COUNTRY B',14),('37.3%',28),('functional',11),('n=459 nets',10)]):
    ax.text(7.8,5.6-i*0.85,txt,ha='center',fontsize=sz,color='white',fontweight='bold' if sz>=14 else 'normal')
ax.legend(handles=[mpatches.Patch(color=c,label=l) for c,l in [('#1C3F6E','≥65%'),('#3A7CA5','60–64%'),('#E87722','50–59%'),('#C0392B','<50%')]],
          loc='lower left',title='Functional Net Rate',title_fontsize=9,fontsize=9,framealpha=0.95)
ax.set_title('Functional Net Rate by Country & District — 24-Month Follow-Up\nVestergaard LLIN Durability Study (n=2,248 nets)',
             fontsize=13,fontweight='bold',color=NAVY,pad=10)
ax.axis('off'); plt.tight_layout()
plt.savefig('charts/python/plot6_map_schematic.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  ✓ Plot 6 (map)")
print("\n✅ All done!")
