

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import kendalltau
import argparse
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import time



@dataclass
class QuadtreeNode:
    points: List[int]
    ranking: Optional[List[int]] = None
    children: Optional[List['QuadtreeNode']] = None
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    def is_leaf(self): return not self.children


class ShiftedQuadtree:
    def __init__(self, data: np.ndarray, max_leaf_size: int = 1):
        self.data = data
        self.n, self.d = data.shape
        self.max_leaf_size = max_leaf_size

    def build(self):
        all_indices = list(range(self.n))
        minb, maxb = np.min(self.data, axis=0), np.max(self.data, axis=0)
        return self._rec(all_indices, minb, maxb)

    def _rec(self, idxs, minb, maxb):
        if len(idxs) <= self.max_leaf_size:
            return QuadtreeNode(points=idxs, bounds=(minb, maxb))
        mid = (minb + maxb) / 2
        kids = []
        for mask in range(2 ** self.d):
            subidx, submin, submax = [], minb.copy(), maxb.copy()
            for dim in range(self.d):
                if (mask >> dim) & 1: submin[dim] = mid[dim]
                else: submax[dim] = mid[dim]
            for i in idxs:
                p = self.data[i]
                if np.all(p >= submin) and np.all(p <= submax):
                    subidx.append(i)
            if subidx:
                kids.append(self._rec(subidx, submin, submax))
        return QuadtreeNode(points=idxs, children=kids, bounds=(minb, maxb))

# =========================
#  Linear Ranking Optimizer
# =========================

class LinearRankingOptimizer:
    def __init__(self, data: np.ndarray, ranking: np.ndarray, top_k: Optional[int] = None,
                 fast_merge: bool = False):
        self.data = np.asarray(data)
        self.ranking = np.asarray(ranking)
        self.n, self.d = self.data.shape
        assert self.ranking.ndim == 1 and len(self.ranking) == self.n
        self.top_k = top_k if top_k else self.n
        self.fast_merge = fast_merge

        # Reuse one Gurobi environment
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

    # ---------- LP feasibility (time-limited) ----------
    def check_feasibility_lp(self, order, eps=1e-3):
        try:
            X, d = self.data, self.data.shape[1]
            m = gp.Model("feas", env=self.env)
            m.Params.OutputFlag = 0
            m.Params.TimeLimit = 0.05
            m.Params.Presolve, m.Params.Threads = 1, 1
            w = m.addVars(d, lb=0)
            for i in range(len(order) - 1):
                i1, i2 = order[i], order[i+1]
                m.addConstr(gp.quicksum(w[k]*(X[i1,k]-X[i2,k]) for k in range(d)) >= eps)
            m.addConstr(gp.quicksum(w[k] for k in range(d)) == 1)
            m.optimize()
            return m.status == GRB.OPTIMAL
        except Exception as e:
            print("LP err:", e)
            return False

    # ---------- Kendall distance (vectorized) ----------
    def kendall_tau_distance(self, r1, r2):
        common = sorted(set(r1) & set(r2))
        if len(common) <= 1: return 0.0
        p1 = {v:i for i,v in enumerate(r1) if v in common}
        p2 = {v:i for i,v in enumerate(r2) if v in common}
        a, b = [p1[v] for v in common], [p2[v] for v in common]
        tau, _ = kendalltau(a, b)
        return 0.5*(1-(tau or 0))

    def compute_ranking_error(self, r, ref):
        order = [i for i in np.argsort(ref) if i in set(r)]
        return self.kendall_tau_distance(r, order)

    # ---------- Greedy merge (uses --fast_merge policy) ----------
    def greedy_merge(self, L1, L2, ref):
        res = L1.copy()
        for it in L2:
            if it in res: continue
            best_r, best_err = None, float('inf')
            
            for pos in range(len(res)+1):
                cand = res[:pos]+[it]+res[pos:]
                err = self.compute_ranking_error(cand, ref)
                if self.fast_merge:  # skip LPs during merge
                    if err < best_err: best_err, best_r = err, cand
                else:
                    if err <= best_err and self.check_feasibility_lp(cand):
                        best_err, best_r = err, cand
            res = best_r if best_r else res+[it]
        return res, self.compute_ranking_error(res, ref)

    def greedy_merge_window(self, base_order, to_insert, ref, window_size: int = 25):
        """Insert items by only probing a small window around their ref rank."""
        res = base_order.copy()
        pos_ref = {i: r for r, i in enumerate(np.argsort(ref))}  # ref rank positions
        for it in to_insert:
            if it in res: continue
            # Target position based on reference ranking
            t = pos_ref[it]
            # Clamp window to current list bounds
            left = max(0, min(len(res), t) - window_size)
            right = min(len(res), max(0, min(len(res), t) + window_size))
            best_r, best_err = None, float('inf')
            # Probe only within [left, right]
            for pos in range(left, right+1):
                cand = res[:pos]+[it]+res[pos:]
                err = self.compute_ranking_error(cand, ref)
                if self.fast_merge:
                    if err < best_err: best_err, best_r = err, cand
                else:
                    if err <= best_err and self.check_feasibility_lp(cand):
                        best_err, best_r = err, cand
            res = best_r if best_r else res+[it]
        return res

    def merge_rankings(self, L1, L2, ref):
        m1,e1=self.greedy_merge(L1,L2,ref)
        m2,e2=self.greedy_merge(L2,L1,ref)
        return m1 if e1<=e2 else m2

    def bottom_up_merge(self,node,ref):
        if node.is_leaf(): return sorted(node.points,key=lambda x:ref[x])
        ch=[self.bottom_up_merge(c,ref) for c in node.children]
        res=ch[0]
        for c in ch[1:]: res=self.merge_rankings(res,c,ref)
        node.ranking=res
        return res

    
    def extract_weights(self, ranking):
        X,d=self.data,self.data.shape[1]
        m=gp.Model("weights",env=self.env)
        m.Params.OutputFlag=0
        m.Params.Presolve,m.Params.Threads,m.Params.TimeLimit=1,1,5
        w=m.addVars(d,lb=0)
        eps=1e-3
        for i in range(len(ranking)-1):
            i1,i2=ranking[i],ranking[i+1]
            m.addConstr(gp.quicksum(w[k]*(X[i1,k]-X[i2,k]) for k in range(d))>=eps)
        m.addConstr(gp.quicksum(w[k] for k in range(d))==1)
        s=m.addVar(lb=0)
        for i in range(len(ranking)-1):
            i1,i2=ranking[i],ranking[i+1]
            m.addConstr(gp.quicksum(w[k]*(X[i1,k]-X[i2,k]) for k in range(d))>=s)
        m.setObjective(s,GRB.MAXIMIZE)
        m.optimize()
        if m.status==GRB.OPTIMAL: return np.array([w[k].X for k in range(d)])
        print("warn: fallback weights"); return np.ones(d)/d

   
    def optimize(self, tree_scope: str = "all", leaf_size: int = 16,
                 insert_mode: str = "baseline", window_size: int = 25):
        """
        tree_scope: "all" (build on n) or "topk" (build on top-k, then insert rest)
        insert_mode (for tree_scope='topk'):
            - 'baseline': greedy insert (slowest)
            - 'window':   probe only ±window_size positions per item
            - 'score':    extract weights from top-k order, score all items, sort by score
        """
        if tree_scope not in ("all","topk"):
            raise ValueError("tree_scope must be 'all' or 'topk'")
        if insert_mode not in ("baseline","window","score"):
            raise ValueError("insert_mode must be 'baseline','window','score'")

        if tree_scope == "all":
            print("Building quadtree on ALL points...")
            tree=ShiftedQuadtree(self.data,max_leaf_size=leaf_size)
            root=tree.build()
            print("Merging...")
            final=self.bottom_up_merge(root,self.ranking)

        else:
            # --- Build/merge on top-k only (local view), then insert remaining globally ---
            print("Building quadtree on TOP-K points...")
            topk_idx = np.argsort(self.ranking)[:self.top_k]              # global ids of top-k
            X_top = self.data[topk_idx]                                    # (TK, d)
            ref_top = self.ranking[topk_idx]                               # local ref ranks
            tree=ShiftedQuadtree(X_top,max_leaf_size=leaf_size)
            root=tree.build()
            print("Merging (top-k only)...")
            local_order = self.bottom_up_merge(root, ref_top)              # local ids 0..TK-1
            topk_global_order = [int(topk_idx[i]) for i in local_order]    # map to global

            remaining = [i for i in range(self.n) if i not in set(topk_global_order)]
            print(f"Remaining to insert: {len(remaining)} (mode={insert_mode})")

            if insert_mode == "baseline":
                final, _ = self.greedy_merge(topk_global_order, remaining, self.ranking)

            elif insert_mode == "window":
                final = self.greedy_merge_window(topk_global_order, remaining,
                                                 self.ranking, window_size=window_size)

            else:  # insert_mode == 'score'
                # 1) Extract weights from the top-k order only
                print("Extracting provisional weights from top-k...")
                w_top = self.extract_weights(topk_global_order)
                # 2) Score everyone once
                scores = self.data @ w_top
                # 3) Final = sort all ids by score descending (fastest)
                final = list(np.argsort(-scores))

        print("Extracting final weights...")
        w=self.extract_weights(final)
        err=self.kendall_tau_distance(final,list(np.argsort(self.ranking)[:self.top_k]))
        print(f"Final Kendall distance (vs top-k ref) = {err:.4f}")
        l2=math.sqrt(sum(v*v for v in w)) or 1
        return np.array([v/l2 for v in w]),final


def _normalize_tiers(t):
    if isinstance(t,set):return[sorted(list(t))]
    if isinstance(t,list):
        if not t:return[]
        if all(isinstance(x,int) for x in t):return[[i]for i in t]
        if all(isinstance(x,list)for x in t):return[ list(x) for x in t]
    raise ValueError

def lppp_diagnostics_counts(points,topk,alpha,strict=False,epsilon=1e-4,tol=1e-12):
    n,d=len(points),len(points[0])
    scores=[sum(alpha[k]*points[i][k]for k in range(d))for i in range(n)]
    tiers=_normalize_tiers(topk)
    top=[i for ti in tiers for i in ti]; topS=set(top)
    outs=[i for i in range(n) if i not in topS]
    cnt,inv=0.0,0;seen=[]
    for ti in tiers:
        for t in ti:
            for c in seen:
                gap=scores[t]-scores[c]
                if gap>0 and gap+epsilon>tol:inv+=1;cnt+=gap
        seen+=ti
    pair,outsAny=0,set()
    for o in outs:
        so=scores[o];v=False
        for t in top:
            gap=so-scores[t]
            if gap>0 and gap+epsilon>tol:pair+=1;cnt+=gap;v=True
        if v:outsAny.add(o)
    rank=sorted(range(n),key=lambda i:scores[i],reverse=True)
    posS={i:r for r,i in enumerate(rank)}
    posG={i:r for r,i in enumerate(top)}
    absd=[abs(posS[i]-posG[i])for i in top]
    return {"pp":float(cnt),"topk_inversions_count":inv,
            "outside_pair_violations":pair,
            "outsiders_with_any_violation":len(outsAny),
            "max_abs_rank_diff":max(absd)if absd else 0,
            "sum_abs_rank_diff":int(sum(absd)) if absd else 0}


def get_nba_dataset():
    df=pd.read_csv('./data/per.csv',skiprows=1,header=0).dropna()
    df=df[df.columns[1:9]]
    sc=MinMaxScaler()
    dfS=pd.DataFrame(sc.fit_transform(df),columns=df.columns)
    return dfS.values

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",default="nba")
    p.add_argument("--items",type=int,default=1000)
    p.add_argument("--tk",type=int,default=5)
    p.add_argument("--seed",type=int,default=10)
    p.add_argument("--verbose",type=int,default=0)
    p.add_argument("--fast_merge",type=int,default=1,
                   help="If 1, skip LP checks during insertion")
    p.add_argument("--tree_scope",choices=["all","topk"],default="topk",
                   help="Build quadtree on all n points or only top-k, then insert the rest")
    p.add_argument("--leaf_size",type=int,default=16,
                   help="Quadtree leaf size (larger => fewer nodes)")
    p.add_argument("--insert_mode",choices=["baseline","window","score"],default="score",
                   help="When tree_scope=topk: how to insert remaining points")
    p.add_argument("--window_size",type=int,default=25,
                   help="Window size for insert_mode=window")
    args=p.parse_args()
    random.seed(args.seed)

    if args.dataset=="nba":
        pts=get_nba_dataset()[:args.items]
        rank=np.arange(args.items)                         # simple rank vector: 0 best, 1 next, ...
        topk=[[i] for i in range(args.tk)]                # for diagnostics

        print("insert mode", args.insert_mode)
        start=time.time()
        opt=LinearRankingOptimizer(pts,rank,args.tk,fast_merge=bool(args.fast_merge))
        alpha,final=opt.optimize(tree_scope=args.tree_scope,
                                 leaf_size=args.leaf_size,
                                 insert_mode=args.insert_mode,
                                 window_size=args.window_size)
        print(f"Runtime={time.time()-start:.2f}s")
        summ=lppp_diagnostics_counts(pts.tolist(),topk,alpha)
        print("summary",summ)
