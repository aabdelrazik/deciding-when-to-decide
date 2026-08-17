import os
"""Per-subject-summed BIC across every fitted model, by horizon and family."""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.append(os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.utils.plotting import compute_metrics

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
rows=[]; missing=[]
for f in sorted(glob.glob(f"{R}/data/simulation_configs/simulation_params_*.py")):
    ns={"__file__":f}; exec(open(f).read(),ns); o=ns["OVERRIDES"]
    base=os.path.basename(f)[len("simulation_params_"):-3]
    commit=base.endswith("_commit"); task=base[:-7] if commit else base
    sub="POMDP_commit" if commit else "POMDP"
    fh=o.get("FIT_HORIZON",["short"]); parts=[fh[0]] if len(fh)==1 else []
    p=os.path.join(R,"data",sub,task,"de",*parts,"results.pkl")
    fam="commit" if commit else "full"
    if not os.path.exists(p): missing.append((task,fam)); continue
    df=pd.read_pickle(p); k=len(o["PARAM_RANGES"]); tot=0.0; n=0
    for _,r in df.iterrows():
        dd=r["data_dict_of_lists"]; hs=dd.values() if isinstance(dd,dict) else dd
        nobs=sum(len(s) for x in hs for s in x["draw_yellow_blue_action_outcome"].values)
        if nobs==0: continue
        tot+=compute_metrics(r["after_lls_ga"],k,nobs)["BIC"]; n+=1
    rows.append(dict(task=task,fam=fam,horizon=task[0],k=k,BIC=tot,n=n,
                     forgetting=("gamma" in o["PARAM_RANGES"])))
d=pd.DataFrame(rows)
for fam in ("full","commit"):
    for h,label in (("S","SHORT"),("L","LONG"),("C","COMBINED")):
        s=d[(d.fam==fam)&(d.horizon==h)].sort_values("BIC")
        if not len(s): continue
        best=s.iloc[0]
        print(f"\n=== {label} / {fam}  ({len(s)} models) ===")
        print(f"  {'model':16}{'k':>3}{'BIC':>12}{'dBIC':>10}   type")
        for _,r in s.head(6).iterrows():
            print(f"  {r.task:16}{r.k:>3}{r.BIC:>12.1f}{r.BIC-best.BIC:>10.1f}   {'forgetting' if r.forgetting else ''}")
        print(f"  -> WINNER: {best.task} (k={best['k']}, BIC={best.BIC:.1f})")
if missing:
    print(f"\n  NOT YET FITTED ({len(missing)}): " + ", ".join(f"{t}[{f}]" for t,f in missing))
d.to_csv(f"{R}/BIC/model_comparison_viper.csv", index=False)
