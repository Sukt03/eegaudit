"""Independent categorical score reimplementation and grouped-target challenge."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/AGENT_LOOP_20261120/results"


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def log_probs(scores, tau):
    logits = scores / tau
    return logits - logsumexp(logits, axis=1, keepdims=True)


def retrieval_gain(lp, ids):
    return (np.log(lp.shape[1]) + lp[np.arange(len(ids)), ids]) / np.log(2)


def loo(scores):
    ids = np.arange(len(scores))
    all_lp = np.stack([log_probs(scores,t) for t in np.logspace(-2.5,0,12)])
    correct = all_lp[:,ids,ids]
    best = np.argmax(correct.sum(axis=1,keepdims=True)-correct,axis=0)
    return all_lp[best,ids]


def partitions(candidates):
    labels = KMeans(n_clusters=5,n_init=10,random_state=20261120).fit_predict(candidates)
    return {"embedding_clusters": labels, "permuted_clusters":np.random.default_rng(20261120).permutation(labels)}


def decompose(lp, ids, labels):
    counts = np.bincount(labels)
    group = labels[ids]
    group_lp = np.stack([logsumexp(lp[:,labels==i],axis=1) for i in range(len(counts))],axis=1)
    hit = group_lp[np.arange(len(ids)),group]
    coarse = (hit-np.log(counts[group]/len(labels)))/np.log(2)
    fine = (lp[np.arange(len(ids)),ids]-hit+np.log(counts[group]))/np.log(2)
    all_gain = retrieval_gain(lp,ids)
    err = float(np.max(np.abs(coarse+fine-all_gain)))
    assert err < 1e-10
    return dict(coarse=coarse.mean().item(), within=fine.mean().item(), total=all_gain.mean().item(),max_chain_error=err)


def image():
    d = np.load(OUT/"i1/i1_sub-01_embeddings.npz")
    candidates = unit(d["candidates"])
    groups = partitions(candidates)
    result = {}
    for cond, features in (("eeg","full"),("noise","noise_full")):
        scores = unit(d[features]) @ candidates.T
        lp = loo(scores)
        result[cond] = {k:decompose(lp,np.arange(len(lp)),v) for k,v in groups.items()}
    historical = json.loads((OUT/"i1/i1_sub-01.json").read_text())
    errors = {arm:abs(result[arm]["embedding_clusters"]["total"]-ref) for arm,ref in (("eeg",historical["rows"][0]["target_yield"]),("noise",historical["target_noise"]))}
    assert max(errors.values()) < 1e-6
    return dict(arms=result, reproduction_errors=errors, reference_record="i1/i1_sub-01.json", group_sizes=np.bincount(groups["embedding_clusters"]).tolist())


def text(task, purge=False):
    sys.path.insert(0,str(ROOT/"experiments/track_text"))
    from pilot_retrieval import load_task,TASKS,embed_sentences,feat
    data = ROOT/"data_raw/zuco"
    sentences, split, records, subjects = load_task(str(data/"eeg2text_pickles"/TASKS[task]),task)
    train,dev,test = [[i for i,s in split.items() if s==name] for name in ("train","dev","test")]
    test.sort()
    removed_train,removed_dev=[],[]
    if purge:
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from zuco_sentence_identity import key
        test_contents={key(sentences[i]) for i in test}
        removed_dev=[i for i in dev if key(sentences[i]) in test_contents]
        dev=[i for i in dev if i not in removed_dev]
        held=test_contents | {key(sentences[i]) for i in dev}
        removed_train=[i for i in train if key(sentences[i]) in held]
        train=[i for i in train if i not in removed_train]
        assert not held.intersection(key(sentences[i]) for i in train)
    enc,name = embed_sentences([sentences[i] for i in train+dev+test],str(data/"minilm"))
    assert "minilm" in name.lower()
    enc = unit(enc)
    train_y = dict(zip(train,enc[:len(train)]))
    dev_y = enc[len(train):len(train)+len(dev)]
    test_y = enc[-len(test):]
    groups = partitions(test_y)
    def features(keys):
        return np.stack([feat(*records[key],"all","mean") for key in keys])
    tr = [k for k in records if k[1] in train_y]
    dv = [k for k in records if k[1] in dev]
    X = features(tr)
    mu,sd = X.mean(0),X.std(0)+1e-8
    X = (X-mu)/sd
    y = np.stack([train_y[k[1]] for k in tr])
    Xdev = (features(dv)-mu)/sd
    truth = np.array([dev.index(k[1]) for k in dv])
    best = -1
    for alpha in (10,100,1000,10000,100000):
        fit = Ridge(alpha=alpha).fit(X,y)
        scores = unit(fit.predict(Xdev)) @ dev_y.T
        success = float(np.mean(scores.argmax(1)==truth))
        if success>best:
            model,best,best_alpha,dev_scores = fit,success,alpha,scores
    tau = max((.01,.02,.05,.1,.2,.5,1.), key=lambda t:retrieval_gain(log_probs(dev_scores,t),truth).mean())
    historical = json.loads((ROOT/f"results/loop/x6_text_bottleneck/x6_{task}.json").read_text())
    full, = [r for r in historical["curve"] if r["basis"]=="pca" and r["L"]==840]
    result = {}
    for subject in subjects:
        te = [(subject,i) for i in test if (subject,i) in records]
        if len(te)<5:
            continue
        truth = np.array([test.index(k[1]) for k in te])
        lp = log_probs(unit(model.predict((features(te)-mu)/sd))@test_y.T,tau)
        result[subject]={k:decompose(lp,truth,v) for k,v in groups.items()}
        result[subject]["original_error"]=abs(float(retrieval_gain(lp,truth).mean())-full["per_participant"][subject]["eeg"]["nce"])
    assert set(result)==set(full["per_participant"])
    return dict(participants=result,N=len(test),removed_train=removed_train,removed_dev=removed_dev,purged=purge,alpha=best_alpha,tau=tau,original_alpha=full["alpha"],original_tau=full["tau"],
                group_sizes=np.bincount(groups["embedding_clusters"]).tolist(),
                reproduction_pass=None if purge else max(r["original_error"] for r in result.values())<1e-4 and best_alpha==full["alpha"] and tau==full["tau"])


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("track",choices=["image","SR","NR2"])
    ap.add_argument("--purge",action="store_true")
    args=ap.parse_args()
    track=args.track
    result=image() if track=="image" else text(track,args.purge)
    result.update(allocation=os.getenv("SLURM_JOB_ID"),step=os.getenv("SLURM_STEP_ID"))
    suffix="_purged" if args.purge else ""
    path=OUT/f"resolution_{track}{suffix}.json"
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)
    print(path,flush=True)
