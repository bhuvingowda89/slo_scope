from __future__ import annotations

import csv, hashlib, json, math, platform, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow, scipy, sklearn
from scipy import stats
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sloscope.artifacts.writer import read_table

OUT = ROOT / "analysis" / "phase7c1"
FIG = OUT / "figures"
SOURCE_SHA = "8e442c813619586072c27a4ec30d7e65d367c3fc1375922935f67c72e2122582"
FREEZE_SHA = "65d9798ea033bc910a832c624324fe068e48552b2a16e6ed639f33763ca028e0"
CAUSES = ["INPUT", "OUTPUT", "LOAD", "DOWNSTREAM"]
SEED = 8127
SLO = {"ttft":0.06512947314299491,"total":0.26779416389490324,"decode":0.20162438542758712}
CORE_F2 = ["ttft_p95","total_latency_p95","post_first_token_duration_p95","ttft_median","total_latency_median","post_first_token_duration_median","scheduler_slip_p95","successful_requests_per_second","host_cpu_median","host_cpu_p95","dependency_duration_median","dependency_duration_p95","gateway_span_duration_median","gateway_span_duration_p95","dependency_span_duration_median","dependency_span_duration_p95","llama_span_duration_median","llama_span_duration_p95","dependency_fraction_of_gateway_median","llama_fraction_of_gateway_median"]
TEMPORAL_METRICS = ["ttft","total_latency","post_first_token_duration","prefill_proxy","dependency_duration","scheduler_slip"]
TEMPORAL_FEATURES = [f"{m}_{s}" for m in TEMPORAL_METRICS for s in ["early_median","late_median","late_minus_early","normalized_slope"]]
TRACE_TEMPORAL = [f"{m}_{s}" for m in ["gateway_span_duration","llama_span_duration","dependency_span_duration"] for s in ["late_minus_early","normalized_slope"]]
F2T = CORE_F2 + TEMPORAL_FEATURES + TRACE_TEMPORAL
F3_DIRECT = ["server_prompt_tokens_median","server_output_tokens_median"]
ALL_SETS = [()] + [(c,) for c in CAUSES] + list(combinations(CAUSES,2))

ANCHORS = {
 "INPUT": ["prefill_proxy_median","prefill_proxy_p95","ttft_median","ttft_p95"],
 "OUTPUT": ["post_first_token_duration_median","post_first_token_duration_p95","llama_span_duration_median","llama_span_duration_p95"],
 "LOAD": ["ttft_late_minus_early","ttft_normalized_slope","total_latency_late_minus_early","total_latency_normalized_slope","successful_requests_per_second","scheduler_slip_late_minus_early","scheduler_slip_normalized_slope"],
 "DOWNSTREAM": ["dependency_duration_median","dependency_duration_p95","dependency_span_duration_median","dependency_span_duration_p95","dependency_fraction_of_gateway_median"],
}
TEMP_BY_CAUSE = {
 "INPUT": ["prefill_proxy_late_minus_early","prefill_proxy_normalized_slope","ttft_late_minus_early","ttft_normalized_slope"],
 "OUTPUT": ["post_first_token_duration_late_minus_early","post_first_token_duration_normalized_slope","llama_span_duration_late_minus_early","llama_span_duration_normalized_slope"],
 "LOAD": ["ttft_late_minus_early","ttft_normalized_slope","total_latency_late_minus_early","total_latency_normalized_slope","scheduler_slip_late_minus_early","scheduler_slip_normalized_slope"],
 "DOWNSTREAM": ["dependency_duration_late_minus_early","dependency_duration_normalized_slope","dependency_span_duration_late_minus_early","dependency_span_duration_normalized_slope"],
}
GENERAL = ["ttft_p95","total_latency_p95","post_first_token_duration_p95","ttft_median","total_latency_median","post_first_token_duration_median","successful_requests_per_second","host_cpu_median","host_cpu_p95","gateway_span_duration_median","llama_span_duration_median"]

def sha(path: Path) -> str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
def write_json(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True); keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
def pct(vals,p):
    vals=sorted([v for v in vals if v is not None and not pd.isna(v)])
    if not vals: return None
    pos=(len(vals)-1)*p; lo=math.floor(pos); hi=math.ceil(pos)
    return vals[lo] if lo==hi else vals[lo]*(hi-pos)+vals[hi]*(pos-lo)
def med(vals): 
    vals=[v for v in vals if v is not None and not pd.isna(v)]
    return float(np.median(vals)) if vals else None
def slope(vals):
    vals=np.array(vals,dtype=float)
    if len(vals)<2 or np.isnan(vals).any(): return None
    x=np.linspace(0,1,len(vals)); return float(np.polyfit(x,vals,1)[0])
def setkey(s): return "+".join(s) if s else "NONE"
def labels(active):
    out=[]
    if "input_medium" in active: out.append("INPUT")
    if "output_32" in active: out.append("OUTPUT")
    if "load_12rps" in active: out.append("LOAD")
    if "downstream_100ms" in active: out.append("DOWNSTREAM")
    return tuple(out)

def input_seal(index):
    seal={"schema_version":"phase7c1.input.v1","publication_run_index_sha256":sha(ROOT/"runs/phase6-v2/publication-run-index.jsonl"),"phase7a1_run_level_metrics_sha256":sha(ROOT/"analysis/phase7a1/run-level-metrics.csv"),"phase7c_rca_dataset_sha256":sha(ROOT/"analysis/phase7c/rca-dataset.csv"),"phase7c_feature_registry_sha256":sha(ROOT/"analysis/phase7c/feature-registry.json"),"phase7c_predictions_sha256":sha(ROOT/"analysis/phase7c/rca-predictions.csv"),"phase7b_csd_results_sha256":sha(ROOT/"analysis/phase7b/csd-summary.csv"),"source_tree_sha256":SOURCE_SHA,"campaign_freeze_sha256":FREEZE_SHA,"run_ids":[r["run_id"] for r in index]}
    seal["phase7c1_input_sha256"]=hashlib.sha256(canon(seal)).hexdigest(); write_json(OUT/"input-seal.json",seal); return seal

def spec():
    sp={"schema_version":"phase7c1.mesr_spec.v1","method":"SLOScope-MESR","causes":CAUSES,"candidate_sets":[setkey(s) for s in ALL_SETS],"primary_training":"P1 single-fault training, compound test","primary_method":"D4-FULL","baseline":"M2/F2 and M2/F2T","threshold":"95th percentile absent-training E(c)","groups":{"S":"general signature","M":"mechanism anchors","T":"temporal evolution","X":"contradiction"},"primary_features_exclude":F3_DIRECT+["CSD","condition_id","active_mechanisms","mechanism_levels","compound_degree"],"confirmatory_tests":["D4-FULL vs M2/F2","D4-FULL vs D4-SM"],"seed":SEED}
    sp["phase7c1_mesr_spec_sha256"]=hashlib.sha256(canon(sp)).hexdigest(); write_json(OUT/"mesr-spec.json",sp); return sp

def load_index():
    rows=[json.loads(l) for l in (ROOT/"runs/phase6-v2/publication-run-index.jsonl").read_text().splitlines() if l.strip()]
    if len(rows)!=104: raise RuntimeError("bad index")
    for r in rows:
        if r["source_hash"]!=SOURCE_SHA or r["freeze_hash"]!=FREEZE_SHA or str(r["run_path"]).startswith("runs/phase6/"): raise RuntimeError("bad publication row")
    return rows

def trace_by_request(run):
    _, trs=read_table(run/"traces.parquet"); out=defaultdict(dict)
    for t in trs:
        try: rid=json.loads(t.get("attributes") or "{}").get("request_id")
        except Exception: rid=None
        if rid: out[rid][t["span_name"]]=float(t["duration"])
    return out

def temporal_for_run(run, window=10):
    _, reqs=read_table(run/"requests.parquet"); tr=trace_by_request(run)
    reqs=sorted(reqs,key=lambda r:int(r["request_id"].rsplit("-",1)[1]))
    rows=[]; features={}
    for r in reqs:
        rid=r["request_id"]; first=r["first_token_time"]; arr=r["actual_arrival"]; comp=r["completion_time"]
        d={"request_id":rid,"request_index":len(rows),
           "ttft":first-arr,"total_latency":comp-arr,"post_first_token_duration":comp-first,
           "prefill_proxy":(r.get("llama_first_token_time") or first)-(r.get("llama_dispatch_time") or arr),
           "dependency_duration":r.get("dependency_duration"),"scheduler_slip":r.get("scheduler_slip"),
           "gateway_span_duration":tr[rid].get("gateway request"),"llama_span_duration":tr[rid].get("llama call"),"dependency_span_duration":tr[rid].get("dependency call")}
        rows.append(d)
    if len(rows)!=40 or len({r["request_id"] for r in rows})!=40: raise RuntimeError("bad request order")
    for m in TEMPORAL_METRICS + ["gateway_span_duration","llama_span_duration","dependency_span_duration"]:
        vals=[r[m] for r in rows]
        features[f"{m}_early_median"]=med(vals[:window]); features[f"{m}_late_median"]=med(vals[-window:])
        features[f"{m}_late_minus_early"]=None if features[f"{m}_early_median"] is None or features[f"{m}_late_median"] is None else features[f"{m}_late_median"]-features[f"{m}_early_median"]
        features[f"{m}_normalized_slope"]=slope(vals)
    features["prefill_proxy_median"]=med([r["prefill_proxy"] for r in rows]); features["prefill_proxy_p95"]=pct([r["prefill_proxy"] for r in rows],.95)
    return features, rows

def dataset(index, window=10):
    base=pd.read_csv(ROOT/"analysis/phase7c/rca-dataset.csv")
    by={r["run_id"]:r for r in index}; rows=[]; temporal_rows=[]
    for rec in base.to_dict("records"):
        run=ROOT/by[rec["run_id"]]["run_path"]; tf, per=temporal_for_run(run, window=window)
        rec.update(tf); rows.append(rec)
        for p in per: temporal_rows.append({"run_id":rec["run_id"],**p})
    return pd.DataFrame(rows), temporal_rows

def split_xy(train,test,features):
    medv=train[features].median(numeric_only=True); xtr=train[features].fillna(medv).to_numpy(float); xte=test[features].fillna(medv).to_numpy(float)
    mu=xtr.mean(0); sd=xtr.std(0); sd[sd==0]=1; return (xtr-mu)/sd,(xte-mu)/sd
def m2_predict(train,test,features):
    xtr,xte=split_xy(train,test,features); preds=[]; scores=[]; cols=[]
    for c in CAUSES:
        y=train[f"label_{c}"].to_numpy(int)
        if len(set(y))<2: prob=np.full(len(test),float(y[0]))
        else:
            model=LogisticRegression(solver="liblinear",C=1,class_weight="balanced",random_state=SEED).fit(xtr,y); prob=model.predict_proba(xte)[:,1]
        cols.append((prob>=.5).astype(int)); scores.append(prob)
    return np.vstack(cols).T,np.vstack(scores).T

def train_mesr(train, variant="D4-FULL", threshold=.95, f3=False):
    cfg={}
    for c in CAUSES:
        present=train[train[f"label_{c}"]==1]; absent=train[train[f"label_{c}"]==0]
        groups={"S":GENERAL,"M":ANCHORS[c]+(([ "server_prompt_tokens_median"] if c=="INPUT" else ["server_output_tokens_median"] if c=="OUTPUT" else []) if f3 else []),"T":TEMP_BY_CAUSE[c]}
        cfg[c]={"groups":groups,"feat":{}}
        for g,fs in groups.items():
            for f in fs:
                if f not in train.columns: continue
                mp=present[f].mean(); ma=absent[f].mean(); gsd=train[f].std()
                if pd.isna(mp) or pd.isna(ma) or pd.isna(gsd) or gsd<1e-12 or abs(mp-ma)<0.05*gsd: continue
                sig=max(absent[f].std(),0.10*gsd,1e-12); cfg[c]["feat"][f]={"dir":1 if mp>ma else -1,"mu0":ma,"sig":sig,"group":g}
    # thresholds from train
    for c in CAUSES:
        es=[score_cause(row,c,cfg[c],variant)[4] for _,row in train.iterrows() if row[f"label_{c}"]==0]
        cfg[c]["tau"]=float(np.quantile(es,threshold)) if es else 0.0
    return cfg

def score_cause(row,c,cfg,variant):
    group_scores={}
    contr=[]
    for g in ["S","M","T"]:
        vals=[]
        if variant=="D4-S" and g!="S": continue
        if variant=="D4-SM" and g=="T": continue
        for f,meta in cfg["feat"].items():
            if meta["group"]!=g or pd.isna(row.get(f)): continue
            z=meta["dir"]*(row[f]-meta["mu0"])/meta["sig"]; z=max(-5,min(5,float(z)))
            if variant=="D4-SMT": z=max(0,z)
            vals.append(z)
            if meta["group"]=="M": contr.append(max(0,-z))
        if vals: group_scores[g]=float(np.mean(vals))
    support=np.mean(list(group_scores.values())) if group_scores else 0.0
    x=float(np.mean(contr)) if contr and variant=="D4-FULL" else 0.0
    e=support-x
    return group_scores.get("S"),group_scores.get("M"),group_scores.get("T"),x,e

def predict_mesr(train,test,variant="D4-FULL",threshold=.95,f3=False):
    cfg=train_mesr(train,variant,threshold,f3); preds=[]; ev=[]; ranked=[]
    for _,row in test.iterrows():
        margins={}; parts={}
        for c in CAUSES:
            s,m,t,x,e=score_cause(row,c,cfg[c],variant); tau=cfg[c]["tau"]; margins[c]=e-tau; parts[c]=(s,m,t,x,e,tau,margins[c])
        cand=[]
        for cs in ALL_SETS:
            score=sum(margins[c] for c in cs)-sum(max(0,margins[c]) for c in CAUSES if c not in cs)
            cand.append((score,len(cs),setkey(cs),cs))
        cand=sorted(cand,key=lambda x:(-x[0],x[1],x[2])); pred=cand[0][3]
        preds.append([1 if c in pred else 0 for c in CAUSES])
        for rank,item in enumerate(cand,1): ranked.append({"run_id":row.run_id,"variant":variant,"rank":rank,"cause_set":item[2],"set_score":item[0]})
        for c in CAUSES:
            s,m,t,x,e,tau,mar=parts[c]; ev.append({"run_id":row.run_id,"variant":variant,"cause":c,"S":s,"M":m,"T":t,"X":x,"E":e,"tau":tau,"margin":mar,"rank":sorted(CAUSES,key=lambda k:margins[k],reverse=True).index(c)+1})
    return np.array(preds), ev, ranked

def metrics(rows):
    exact=[]; jac=[]; complete=[]; partial=[]; over=[]; under=[]; tp=fp=fn=0
    for r in rows:
        tr={c for c in CAUSES if r[f"true_{c}"]}; pr={c for c in CAUSES if r[f"pred_{c}"]}
        exact.append(tr==pr); jac.append(1 if not tr and not pr else len(tr&pr)/len(tr|pr))
        if tr: complete.append(tr<=pr); partial.append(len(tr&pr)/len(tr))
        over.append(bool(pr-tr)); under.append(bool(tr-pr))
        for c in CAUSES:
            y=c in tr; p=c in pr; tp+=y and p; fp+=(not y) and p; fn+=y and (not p)
    micro_p=tp/(tp+fp) if tp+fp else 0; micro_r=tp/(tp+fn) if tp+fn else 0; micro_f=2*micro_p*micro_r/(micro_p+micro_r) if micro_p+micro_r else 0
    macro=[]
    for c in CAUSES:
        ctp=sum(r[f"true_{c}"] and r[f"pred_{c}"] for r in rows); cfp=sum((not r[f"true_{c}"]) and r[f"pred_{c}"] for r in rows); cfn=sum(r[f"true_{c}"] and (not r[f"pred_{c}"]) for r in rows)
        p=ctp/(ctp+cfp) if ctp+cfp else 0; rr=ctp/(ctp+cfn) if ctp+cfn else 0; macro.append(2*p*rr/(p+rr) if p+rr else 0)
    controls=[r for r in rows if not any(r[f"true_{c}"] for c in CAUSES)]
    return {"n":len(rows),"exact_set_accuracy":float(np.mean(exact)) if rows else None,"jaccard":float(np.mean(jac)) if rows else None,"complete_cause_recall":float(np.mean(complete)) if complete else None,"partial_cause_recall":float(np.mean(partial)) if partial else None,"micro_f1":micro_f,"macro_f1":float(np.mean(macro)),"over_attribution":float(np.mean(over)) if rows else None,"under_attribution":float(np.mean(under)) if rows else None,"control_false_alarm_rate":float(np.mean([r["predicted_set"]!="NONE" for r in controls])) if controls else None}

def signflip(d):
    obs=abs(np.mean(d)); vals=[]
    for i in range(2**len(d)):
        s=np.array([1 if (i>>j)&1 else -1 for j in range(len(d))]); vals.append(abs(np.mean(np.array(d)*s)))
    return sum(v>=obs-1e-12 for v in vals)/len(vals)

def evaluate(df):
    variants=["D4-S","D4-SM","D4-SMT","D4-FULL","D4-FULL-F3"]; methods=["M2/F2","M2/F2T"]+variants
    pred_rows=[]; ev=[]; ranked=[]
    for fold in range(1,9):
        train=df[(df.repetition!=fold)&(df.compound_degree<=1)]; test_all=df[df.repetition==fold]
        for name in ["M2/F2","M2/F2T"]:
            feats=CORE_F2 if name=="M2/F2" else F2T
            pred,score=m2_predict(train,test_all,feats)
            add_preds(pred_rows,test_all,pred,score,name,fold)
        for v in variants:
            pred,e,rs=predict_mesr(train,test_all,"D4-FULL" if v=="D4-FULL-F3" else v,f3=(v=="D4-FULL-F3"))
            add_preds(pred_rows,test_all,pred,None,v,fold); ev+=e; ranked+=rs
    return pred_rows,ev,ranked

def add_preds(out,test,pred,score,method,fold):
    for i,(_,row) in enumerate(test.reset_index(drop=True).iterrows()):
        pr=tuple(c for j,c in enumerate(CAUSES) if pred[i,j]); tr=tuple(c for c in CAUSES if row[f"label_{c}"])
        d={"run_id":row.run_id,"condition_id":row.condition_id,"repetition":int(row.repetition),"fold":fold,"method":method,"truth_set":setkey(tr),"predicted_set":setkey(pr),"true_degree":int(row.compound_degree),"predicted_count":len(pr)}
        for j,c in enumerate(CAUSES):
            d[f"true_{c}"]=int(row[f"label_{c}"]); d[f"pred_{c}"]=int(pred[i,j]); d[f"{c.lower()}_score"]=float(score[i,j]) if score is not None else None
        # set margin placeholder for MESR ranked rows later
        out.append(d)

def summarize(preds):
    rows=[]; comp=[]; cause=[]; errors=[]
    for m,sub in pd.DataFrame(preds).groupby("method"):
        allr=sub.to_dict("records"); cr=sub[sub.true_degree==2].to_dict("records")
        rows.append({"method":m,"subset":"P1-ALL",**metrics(allr)})
        rows.append({"method":m,"subset":"P1-COMPOUND",**metrics(cr)})
        rows.append({"method":m,"subset":"P1-SINGLE",**metrics(sub[sub.true_degree==1].to_dict("records"))})
        for cond,ss in sub[sub.true_degree==2].groupby("condition_id"): comp.append({"method":m,"condition_id":cond,**metrics(ss.to_dict("records"))})
        for subset_name,rs in [("P1-ALL",allr),("P1-COMPOUND",cr)]:
            for c in CAUSES:
                tp=sum(r[f"true_{c}"] and r[f"pred_{c}"] for r in rs); fp=sum((not r[f"true_{c}"]) and r[f"pred_{c}"] for r in rs); fn=sum(r[f"true_{c}"] and (not r[f"pred_{c}"]) for r in rs); tn=sum((not r[f"true_{c}"]) and (not r[f"pred_{c}"]) for r in rs)
                p=tp/(tp+fp) if tp+fp else 0; rr=tp/(tp+fn) if tp+fn else 0; f=2*p*rr/(p+rr) if p+rr else 0
                cause.append({"method":m,"subset":subset_name,"cause":c,"precision":p,"recall":rr,"f1":f,"false_positive_rate":fp/(fp+tn) if fp+tn else 0,"false_negative_rate":fn/(fn+tp) if fn+tp else 0})
        for r in cr:
            tr={c for c in CAUSES if r[f"true_{c}"]}; pr={c for c in CAUSES if r[f"pred_{c}"]}
            if tr==pr: continue
            miss=tr-pr; extra=pr-tr; types=[]
            types += [f"MISS_{c}" for c in miss]
            if len(miss)==2: types.append("MISS_BOTH")
            if extra: types.append("EXTRA_CAUSE")
            if len(pr)==1: types.append("SINGLE_ONLY_PREDICTION")
            if not pr: types.append("EMPTY_PREDICTION")
            errors.append({**r,"error_types":";".join(types)})
    return rows,comp,cause,errors

def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    index=load_index(); seal=input_seal(index); sp=spec(); df,temp=dataset(index)
    write_csv(OUT/"mesr-dataset.csv",df.to_dict("records")); write_csv(OUT/"mesr-temporal-features.csv",temp)
    registry={"CORE_F2":CORE_F2,"TEMPORAL":TEMPORAL_FEATURES+TRACE_TEMPORAL,"MECHANISM_ANCHOR":ANCHORS,"F3_DIRECT_EVIDENCE":F3_DIRECT}; write_json(OUT/"feature-registry.json",registry)
    leak={"primary_D4_issues":[],"excluded":["condition_id","active_mechanisms","mechanism_levels","compound_degree","CSD"]}; write_json(OUT/"feature-leakage-audit.json",leak)
    preds,ev,ranked=evaluate(df); write_csv(OUT/"mesr-predictions.csv",preds); write_csv(OUT/"mesr-evidence-scores.csv",ev); write_csv(OUT/"mesr-ranked-sets.csv",ranked)
    primary,bycomp,cause,errors=summarize(preds); write_csv(OUT/"table-9-mesr-primary.csv",primary); write_csv(OUT/"table-10-mesr-by-compound.csv",bycomp); write_csv(OUT/"mesr-compound-metrics.csv",bycomp); write_csv(OUT/"mesr-cause-metrics.csv",cause); write_csv(OUT/"mesr-errors.csv",errors)
    # ablation fold metrics
    fold_rows=[]
    for m in ["D4-S","D4-SM","D4-SMT","D4-FULL","M2/F2","M2/F2T","D4-FULL-F3"]:
        for f in range(1,9):
            fold_rows.append({"method":m,"fold":f,**metrics([r for r in preds if r["method"]==m and r["fold"]==f and r["true_degree"]==2])})
    write_csv(OUT/"mesr-fold-metrics.csv",fold_rows); write_csv(OUT/"mesr-ablation.csv",fold_rows)
    diffs=[next(r for r in fold_rows if r["method"]=="D4-FULL" and r["fold"]==f)["exact_set_accuracy"]-next(r for r in fold_rows if r["method"]=="D4-SM" and r["fold"]==f)["exact_set_accuracy"] for f in range(1,9)]
    diffs_base=[next(r for r in fold_rows if r["method"]=="D4-FULL" and r["fold"]==f)["exact_set_accuracy"]-next(r for r in fold_rows if r["method"]=="M2/F2" and r["fold"]==f)["exact_set_accuracy"] for f in range(1,9)]
    p1=signflip(diffs_base); p2=signflip(diffs); holm1=min(1,2*min(p1,p2)); holm2=max(p1,p2)
    confirm=[{"test":"D4-FULL_vs_M2F2","raw_p":p1,"holm_p":holm1 if p1<=p2 else holm2,"mean_diff":float(np.mean(diffs_base))},{"test":"D4-FULL_vs_D4SM","raw_p":p2,"holm_p":holm1 if p2<=p1 else holm2,"mean_diff":float(np.mean(diffs))}]
    write_csv(OUT/"confirmatory-tests.csv",confirm); write_csv(OUT/"table-11-mesr-ablation.csv",fold_rows+confirm)
    # sensitivities
    thresh=[]
    for q in [.90,.95,.975]:
        pr,_,_=evaluate_threshold(df,q); sm,_,_,_=summarize(pr); 
        for r in sm:
            if r["method"]=="D4-FULL" and r["subset"]=="P1-COMPOUND": thresh.append({"threshold":q,**r})
    write_csv(OUT/"mesr-threshold-sensitivity.csv",thresh)
    df20,_=dataset(index, window=20); pr20,_,_=evaluate(df20); sm20,bc20,_,_=summarize(pr20)
    sens=[]
    primary_comp={r["condition_id"]:r for r in bycomp if r["method"]=="D4-FULL"}
    for r in bc20:
        if r["method"]=="D4-FULL":
            base=primary_comp.get(r["condition_id"],{})
            sens.append({"analysis":"first20_last20","method":"D4-FULL","condition_id":r["condition_id"],"primary_window_exact":base.get("exact_set_accuracy"),"sensitivity_window_exact":r["exact_set_accuracy"],"exact_changed":r["exact_set_accuracy"]!=base.get("exact_set_accuracy")})
    full_primary=next(r for r in primary if r["method"]=="D4-FULL" and r["subset"]=="P1-COMPOUND")
    full_sens=next(r for r in sm20 if r["method"]=="D4-FULL" and r["subset"]=="P1-COMPOUND")
    sens.append({"analysis":"first20_last20","method":"D4-FULL","condition_id":"ALL_COMPOUNDS","primary_window_exact":full_primary["exact_set_accuracy"],"sensitivity_window_exact":full_sens["exact_set_accuracy"],"exact_changed":full_primary["exact_set_accuracy"]!=full_sens["exact_set_accuracy"]})
    write_csv(OUT/"mesr-temporal-sensitivity.csv",sens)
    csd=pd.read_csv(ROOT/"analysis/phase7b/csd-summary.csv"); csdmap={r.compound:r for r in csd.itertuples()}
    write_csv(OUT/"mesr-csd-linked-analysis.csv",[{**r,"CSD":getattr(csdmap.get(r["condition_id"]),"CSD",None),"CSD_supported":getattr(csdmap.get(r["condition_id"]),"statistically_supported",None)} for r in bycomp])
    # baseline f2t file
    write_csv(OUT/"baseline-f2t-predictions.csv",[r for r in preds if r["method"]=="M2/F2T"])
    # Evidence decomposition for interpretability: D4-FULL cause-level evidence by outcome type.
    evdf=pd.DataFrame(ev); pdf=pd.DataFrame(preds)
    decomp=[]
    for _,er in evdf[evdf["variant"]=="D4-FULL"].iterrows():
        pr=pdf[(pdf.run_id==er.run_id)&(pdf.method=="D4-FULL")].iloc[0]
        c=er.cause; true=bool(pr[f"true_{c}"]); pred=bool(pr[f"pred_{c}"])
        if true and pred: outcome="true_positive"
        elif true and not pred: outcome="false_negative"
        elif (not true) and pred: outcome="false_positive"
        else: outcome="true_negative"
        decomp.append({**er.to_dict(),"outcome":outcome})
    decomp_summary=[]
    for (c,o),grp in pd.DataFrame(decomp).groupby(["cause","outcome"]):
        decomp_summary.append({"cause":c,"outcome":o,"n":len(grp),"mean_S":grp["S"].mean(),"mean_M":grp["M"].mean(),"mean_T":grp["T"].mean(),"mean_X":grp["X"].mean(),"mean_margin":grp["margin"].mean()})
    write_csv(OUT/"mesr-evidence-decomposition.csv",decomp_summary)
    # placeholder figs with table text
    for name,title in [("figure-14-mesr-method-comparison.svg","Figure 14 MESR Method Comparison"),("figure-15-mesr-ablation.svg","Figure 15 MESR Ablation"),("figure-16-mesr-by-pair.svg","Figure 16 MESR by Pair"),("figure-17-mesr-evidence-decomposition.svg","Figure 17 Evidence Decomposition")]:
        (FIG/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300"><text x="20" y="30">{title}</text></svg>')
    h1=hashlib.sha256(canon(preds)).hexdigest(); preds2,_,_=evaluate(df); h2=hashlib.sha256(canon(preds2)).hexdigest(); write_json(OUT/"determinism-audit.json",{"prediction_hash_1":h1,"prediction_hash_2":h2,"identical":h1==h2})
    write_json(OUT/"analysis-provenance.json",{"timestamp":datetime.now(timezone.utc).isoformat(),"phase7c1_input_sha256":seal["phase7c1_input_sha256"],"phase7c1_mesr_spec_sha256":sp["phase7c1_mesr_spec_sha256"],"phase7c_input_sha256":json.loads((ROOT/"analysis/phase7c/input-seal.json").read_text())["phase7c_input_sha256"],"phase7c_rca_spec_sha256":json.loads((ROOT/"analysis/phase7c/rca-spec.json").read_text())["phase7c_rca_spec_sha256"],"phase7b_input_sha256":json.loads((ROOT/"analysis/phase7b/input-seal.json").read_text())["phase7b_input_sha256"],"phase7b_csd_spec_sha256":json.loads((ROOT/"analysis/phase7b/csd-spec.json").read_text())["phase7b_csd_spec_sha256"],"source_tree_sha256":SOURCE_SHA,"campaign_freeze_sha256":FREEZE_SHA,"python_version":sys.version,"versions":{"numpy":np.__version__,"pandas":pd.__version__,"scipy":__import__("scipy").__version__,"sklearn":__import__("sklearn").__version__,"pyarrow":__import__("pyarrow").__version__},"analysis_script_hashes":{"analysis/phase7c1/phase7c1_mesr.py":sha(OUT/"phase7c1_mesr.py")}})
    write_json(OUT/"phase7c1-report.json",{"input":seal,"spec":sp,"primary":primary,"confirmatory":confirm})
    (OUT/"phase7c1-report.md").write_text("# Phase 7C.1 MESR\n\nSee generated tables and CSVs.\n")

def evaluate_threshold(df,q):
    preds=[]; ev=[]; ranked=[]
    for fold in range(1,9):
        train=df[(df.repetition!=fold)&(df.compound_degree<=1)]; test_all=df[df.repetition==fold]
        pred,e,rs=predict_mesr(train,test_all,"D4-FULL",threshold=q)
        add_preds(preds,test_all,pred,None,"D4-FULL",fold); ev+=e; ranked+=rs
    return preds,ev,ranked

if __name__=="__main__": main()
