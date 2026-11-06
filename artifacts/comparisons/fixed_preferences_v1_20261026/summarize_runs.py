"""Read-only descriptive comparison; outputs only beside this artifact script."""
from pathlib import Path
import json, statistics as st
from collections import Counter, defaultdict
ROOT=Path(__file__).resolve().parents[3]
PATHS={
'baseline':'experiments/fixed_preferences_v1/runs/baseline/20261025T134921Z-99290f7f',
'25V':'experiments/fixed_preferences_v1/runs/25v/20261025T145836Z-65a997b1',
'25F':'experiments/fixed_preferences_v1/runs/25f/20261026T032217Z-54514d0b'}
PHASES={'pre':(1,15),'insurance':(16,30),'post':(31,45),'tail':(46,48),'all45':(1,45)}
raw={}
out={}
def readrows(path):return [json.loads(line) for line in path.read_text().splitlines() if line]
def summarize(rs, partners):
 if not rs:return {'n':0}
 amounts=[r['transfer_sent'] for r in rs]
 gaps=[r['income_before_transfer']-partners[(r['partner_id'],r['period'])]['income_before_transfer'] for r in rs]
 return {'n':len(rs),'mean':st.mean(amounts),'median':st.median(amounts),'sd':st.stdev(amounts) if len(rs)>1 else None,'min':min(amounts),'max':max(amounts),'zero_count':amounts.count(0),'zero_rate':amounts.count(0)/len(rs),'equalized_count':sum(r['final_income']==partners[(r['partner_id'],r['period'])]['final_income'] for r in rs),'equalized_rate':sum(r['final_income']==partners[(r['partner_id'],r['period'])]['final_income'] for r in rs)/len(rs),'within_one_point_count':sum(abs(r['final_income']-partners[(r['partner_id'],r['period'])]['final_income'])<=1 for r in rs),'mean_fraction_of_equalizing_transfer':st.mean(a/(g/2) for a,g in zip(amounts,gaps)),'amount_counts':dict(sorted(Counter(amounts).items()))}
for treatment,rel in PATHS.items():
 path=ROOT/rel
 rows=readrows(path/'period_records.jsonl');raw[treatment]=rows
 lookup={(r['agent_id'],r['period']):r for r in rows}
 donors=[r for r in rows if r['had_transfer_opportunity']]
 m=json.loads((path/'manifest.json').read_text())
 members={pid:sorted({r['agent_id'] for r in rows if r['pair_id']==pid}) for pid in sorted({r['pair_id'] for r in rows})}
 insurance={}
 if treatment=='25V':insurance={r['agent_id']:r['purchase'] for r in readrows(path/'insurance_decisions.jsonl')}
 elif treatment=='25F':insurance={r['agent_id']:r['insured'] for r in readrows(path/'insurance_assignments.jsonl')}
 else:insurance={a:False for pair in members.values() for a in pair}
 categories={pid:('none','single','both')[sum(insurance[a] for a in agents)] for pid,agents in members.items()}
 phases={phase:summarize([r for r in donors if lo<=r['period']<=hi],lookup) for phase,(lo,hi) in PHASES.items()}
 bypair={str(pid):{'members':agents,'insurance_category':categories[pid],'insured_members':[a for a in agents if insurance[a]],'phase_means':{phase:st.mean(r['transfer_sent'] for r in donors if r['pair_id']==pid and lo<=r['period']<=hi) for phase,(lo,hi) in PHASES.items()}} for pid,agents in members.items()}
 for item in bypair.values():
  item['post_minus_pre']=item['phase_means']['post']-item['phase_means']['pre']
  item['insurance_minus_pre']=item['phase_means']['insurance']-item['phase_means']['pre']
 bycategory={category:{'pairs':[pid for pid in members if categories[pid]==category],'phases':{phase:summarize([r for r in donors if categories[r['pair_id']]==category and lo<=r['period']<=hi],lookup) for phase,(lo,hi) in PHASES.items()}} for category in ('none','single','both')}
 bysingle={kind:{phase:summarize([r for r in donors if categories[r['pair_id']]=='single' and insurance[r['agent_id']]==flag and lo<=r['period']<=hi],lookup) for phase,(lo,hi) in PHASES.items()} for kind,flag in [('insured',True),('uninsured',False)]}
 byagent={str(a):{'pair_id':next(r['pair_id'] for r in rows if r['agent_id']==a),'insurance':insurance[a],'phase_means':{phase:summarize([r for r in donors if r['agent_id']==a and lo<=r['period']<=hi],lookup) for phase,(lo,hi) in PHASES.items()}} for a in sorted(insurance)}
 out[treatment]={'run':rel,'backend':m['backend'],'success':m['success'],'completed_periods':m['completed_periods'],'participants':len(insurance),'pairs':len(members),'insurance_count':sum(insurance.values()),'insurance_rate':sum(insurance.values())/len(insurance),'insurance_by_agent':insurance,'category_counts':dict(Counter(categories.values())),'phases':phases,'by_pair':bypair,'by_category':bycategory,'single_pair_by_insurance_status':bysingle,'by_agent':byagent}
baseindex={(r['agent_id'],r['period']):r for r in raw['baseline']}
for t in ('25V','25F'):
 out[t]['same_pairing_as_baseline']=all(r['pair_id']==baseindex[(r['agent_id'],r['period'])]['pair_id'] and r['partner_id']==baseindex[(r['agent_id'],r['period'])]['partner_id'] for r in raw[t])
 out[t]['same_income_shocks_as_baseline']=all(r['random_income']==baseindex[(r['agent_id'],r['period'])]['random_income'] for r in raw[t])
 out[t]['descriptive_did']={phase:(out[t]['phases'][phase]['mean']-out[t]['phases']['pre']['mean'])-(out['baseline']['phases'][phase]['mean']-out['baseline']['phases']['pre']['mean']) for phase in ('insurance','post')}
 out[t]['single_pair_agent_equal_weight'] = {}
 for status, flag in (('insured', True), ('uninsured', False)):
  agents = sorted(int(agent) for agent, value in out[t]['by_agent'].items()
                  if value['insurance'] == flag and out[t]['by_pair'][str(value['pair_id'])]['insurance_category'] == 'single')
  out[t]['single_pair_agent_equal_weight'][status] = {
   'agents': agents,
   'phase_means': {phase: st.mean(out[t]['by_agent'][str(agent)]['phase_means'][phase]['mean'] for agent in agents)
                   for phase in ('pre', 'insurance', 'post')},
   'individual_post_minus_pre': {str(agent): out[t]['by_agent'][str(agent)]['phase_means']['post']['mean']
                                 - out[t]['by_agent'][str(agent)]['phase_means']['pre']['mean'] for agent in agents},
  }
Path(__file__).with_name('statistics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for t,d in out.items():
 print(t, 'insured',d['insurance_count'],'category_counts',d['category_counts'])
 for phase in ('pre','insurance','post','tail'):
  p=d['phases'][phase]
  print(phase,{k:p[k] for k in ('n','mean','median','zero_rate','equalized_rate','mean_fraction_of_equalizing_transfer')})
 print('pair changes',[(k,round(v['post_minus_pre'],4)) for k,v in d['by_pair'].items()])
 if 'descriptive_did' in d:print('DID',d['descriptive_did'],'samepairs',d['same_pairing_as_baseline'],'sameincome',d['same_income_shocks_as_baseline'])
 for cat,v in d['by_category'].items():
  if v['pairs']:print(cat,v['pairs'],{p:round(v['phases'][p]['mean'],4) for p in ('pre','insurance','post')})
 if d['category_counts'].get('single'):
  print('single member means',{k:{p:round(v[p]['mean'],4) for p in ('pre','insurance','post')} for k,v in d['single_pair_by_insurance_status'].items()})
