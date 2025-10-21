import time
import gurobipy as gp
import numpy as np
import logging
from itertools import combinations
import math
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random
import itertools
import argparse


file_handler = logging.FileHandler('app.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)



def _normalize_tiers(topk_ranking):
    """set -> single tie tier; list -> linear order; list of lists -> tiers as given."""
    def _is_intlike(x):
        try:
            import numpy as _np
            return isinstance(x, (int,)) or isinstance(x, _np.integer)
        except Exception:
            return isinstance(x, int)
    if isinstance(topk_ranking, set):
        return [list(int(i) for i in topk_ranking)]
    elif hasattr(topk_ranking, "__iter__") and all(_is_intlike(x) for x in topk_ranking):
        return [[int(i)] for i in topk_ranking]
    else:
        return [[int(i) for i in tier] for tier in topk_ranking]

def lppp_diagnostics_counts(points, topk_ranking, alpha_unit, *, strict=False, epsilon=1e-4, tol=1e-12):
    """
    Returns:
      {
        "pp": float,                           # raw sum of positive parts (score gaps)
        "topk_inversions_count": int,          # earlier (better) scored worse than later (worse)
        "outside_pair_violations": int,        # count of (outsider, topk) with s[o]-s[t]+margin > tol
        "outsiders_with_any_violation": int,   # number of distinct outsiders with any such violation
        "max_abs_rank_diff": int,              # max |pos_score - pos_given| over top-k items
        "sum_abs_rank_diff": int               # sum |pos_score - pos_given| over top-k items
      }
    """
    n, dims = len(points), len(points[0])
    assert len(alpha_unit) == dims, "alpha dimension mismatch"

    scores = [sum(alpha_unit[d]*points[i][d] for d in range(dims)) for i in range(n)]
    if args.verbose:
        print("scores", scores)

    tiers = _normalize_tiers(topk_ranking)
    ordered_topk = [i for tier in tiers for i in tier]
    topk_set = set(ordered_topk)
    outsiders = [i for i in range(n) if i not in topk_set]

    margin = 0.0 if strict else epsilon

    # ----- existing violation metrics  -----
    count_pp = 0.0
    inversions = 0
    seen = []
    for tier in tiers:
        for t in tier:               # later/worse
            for c in seen:           # earlier/better
                gap = scores[t] - scores[c]
                if gap > 0:
                    if gap + margin > tol:
                        inversions += 1
                        count_pp += gap
        seen.extend(tier)

    pair_viol = 0
    outsider_any = set()
    for o in outsiders:
        so = scores[o]
        violated = False
        for t in ordered_topk:
            gap = so - scores[t]
            if gap > 0:
                if gap + margin > tol:
                    pair_viol += 1
                    count_pp += gap
                    violated = True
        if violated:
            outsider_any.add(o)

    # ----- ranking discrepancy between score ranking and given top-k order -----
    score_ranking = sorted(range(n), key=lambda i: scores[i], reverse=True)
    pos_score = {i: r for r, i in enumerate(score_ranking)}
    pos_given = {i: r for r, i in enumerate(ordered_topk)}  
    if args.verbose:
        print("pos_score", pos_score)
        print("pos_given", pos_given)

    abs_diffs = [abs(pos_score[i] - pos_given[i]) for i in ordered_topk]
    max_abs_rank_diff = max(abs_diffs) if abs_diffs else 0
    sum_abs_rank_diff = int(sum(abs_diffs)) if abs_diffs else 0

    return {
        "pp": count_pp,
        "topk_inversions_count": inversions,
        "outside_pair_violations": pair_viol,
        "outsiders_with_any_violation": len(outsider_any),
        "max_abs_rank_diff": max_abs_rank_diff,
        "sum_abs_rank_diff": sum_abs_rank_diff,
    }


def get_nba_dataset():
    df = pd.read_csv('./data/per.csv', skiprows=1, header=0).dropna()
    df = df[df.columns[1:9]]
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled.values.tolist()


def _scores_from_weights(points_np: np.ndarray, w: np.ndarray) -> np.ndarray:
    return points_np @ w

def _ranking_from_scores(scores: np.ndarray) -> list:
    return np.argsort(-scores).tolist()

def _kendall_topk_error_from_ranking(ranking: list, ground_truth_topk: list) -> int:
    """Pairs inside top-k in wrong order + outsiders above any top-k."""
    ground_truth_set = set(ground_truth_topk)
    pos = {item: p for p, item in enumerate(ranking)}
    inv = 0
    # type1: order inside top-k
    for i in range(len(ground_truth_topk)):
        for j in range(i+1, len(ground_truth_topk)):
            a = ground_truth_topk[i]
            b = ground_truth_topk[j]
            if pos[a] > pos[b]:
                inv += 1
    # type2: outsider outranking top-k
    for p, item in enumerate(ranking):
        if item not in ground_truth_set:
            for gt in ground_truth_topk:
                if pos[gt] > p:
                    inv += 1
    return inv

def _is_ranking_realisable(points_np: np.ndarray, ranking: list, epsilon: float=1e-4):
    
    n, d = points_np.shape
    m = gp.Model()
    m.Params.OutputFlag = 0
    w = m.addMVar(shape=d, lb=epsilon, ub=1000.0)
    for i in range(len(ranking)-1):
        a = ranking[i]
        b = ranking[i+1]
        diff = points_np[a] - points_np[b]
        m.addConstr(w @ diff >= epsilon)
    m.addConstr(w.sum() == 1.0)
    m.setObjective(w.sum(), gp.GRB.MINIMIZE)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        return True, np.array([w[i].X for i in range(d)])
    return False, np.zeros(d)

def greedy_refine_weights(points_np: np.ndarray,
                          initial_w: np.ndarray,
                          ground_truth_topk: list,
                          verbose: bool=False):
    """Greedy adjacent swaps: accept if error improves AND realizable; update weights to the LP solution."""
    w = initial_w.copy()
    scores = _scores_from_weights(points_np, w)
    ranking = _ranking_from_scores(scores)
    best_err = _kendall_topk_error_from_ranking(ranking, ground_truth_topk)
    if verbose:
        print(f"Initial error: {best_err}")

    improved = True
    it = 0
    lp_calls = 0
    while improved:
        improved = False
        it += 1
        for i in range(len(ranking)-1):
            new_r = ranking.copy()
            new_r[i], new_r[i+1] = new_r[i+1], new_r[i]
            new_err = _kendall_topk_error_from_ranking(new_r, ground_truth_topk)
            if new_err >= best_err:
                continue
            # Only now check realizability
            feasible, w_cand = _is_ranking_realisable(points_np, new_r)
            lp_calls += 1
            if not feasible:
                continue
            # accept
            ranking = new_r
            w = w_cand
            best_err = new_err
            if verbose:
                print(f"Iteration {it}: improvement -> error {best_err}")
            improved = True
            break
    if verbose:
        print(f"Reached local minimum after {it} iterations; LP calls: {lp_calls}")
    return w, best_err

def _random_positive_unit_weights(d: int, rng: np.random.RandomState) -> np.ndarray:
    w = np.abs(rng.randn(d))
    s = w.sum()
    if s == 0:
        w = np.ones(d) / d
    else:
        w = w / s
    return w

def random_sampling_best(points_np: np.ndarray,
                         ground_truth_topk: list,
                         samples: int,
                         seed: int=None,
                         verbose: bool=False):
    rng = np.random.RandomState(seed) if seed is not None else np.random
    n, d = points_np.shape
    best_w = None
    best_err = float('inf')
    for _ in range(samples):
        w = _random_positive_unit_weights(d, rng)
        err = _kendall_topk_error_from_ranking(_ranking_from_scores(points_np @ w), ground_truth_topk)
        if err < best_err:
            best_err = err
            best_w = w
    if verbose:
        print(f"Random sampling best error: {best_err}")
    return best_w, best_err


def get_jee_dataset(csv_path: str, drop_cols=None, scale=False):
    if drop_cols is None:
        drop_cols = ["RANK", "BLOCK_ID", "AIR", "REGST_NO", "mark"]

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if "sub_category" in df.columns:
        df = df.drop(columns=["sub_category"])

    if "category" in df.columns:
        cat_map = {"GEN": 0, "OBC-NCL": 1, "SC": 2, "ST": 3}
        df["category"] = (
            df["category"].astype(str).str.strip().str.upper().map(cat_map)
        )

    if "GENDER" in df.columns:
        gender_map = {"M": 0, "F": 1}
        df["GENDER"] = (
            df["GENDER"].astype(str).str.strip().str.upper().map(gender_map)
        )

    if "is_pwd" in df.columns:
        def _to_bin(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, str):
                s = x.strip().lower()
                if s in {"1", "true", "t", "yes", "y"}: return 1.0
                if s in {"0", "false", "f", "no", "n"}: return 0.0
            try:
                return 1.0 if bool(int(x)) else 0.0
            except Exception:
                return 1.0 if bool(x) else 0.0
        df["is_pwd"] = df["is_pwd"].apply(_to_bin)

    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.empty:
        raise ValueError("No numeric columns found after encoding and dropping IDs.")

    if scale:
        scaler = MinMaxScaler()
        encoded_cols = [c for c in ["category", "GENDER", "is_pwd"] if c in num_df.columns]
        cont_cols = [c for c in num_df.columns if c not in encoded_cols]
        if cont_cols:
            num_df[cont_cols] = scaler.fit_transform(num_df[cont_cols].values)

    return num_df.values.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["nba", "jee"], default="nba",
                        help="Choose dataset: nba (default) or jee")
    parser.add_argument("--jee_csv", type=str, default="./data/reserved_blocks_100.csv",
                        help="Path to JEE CSV when --dataset jee")
    parser.add_argument("--jee_scale", type=int, default=0,
                        help="If 1, apply MinMax scaling to JEE features (default 0)")
    parser.add_argument("--items", type=int, default=1000)
    parser.add_argument("--tk", type=int, default=100)
    parser.add_argument("--randomsuffle", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--samples", type=int, default=1000,
                        help="Random samples for initial weights (positive, sum=1)")
    parser.add_argument("--greedy_verbose", type=int, default=0,
                        help="Print greedy refinement progress")

    args = parser.parse_args()

    TK = args.tk
    items = args.items
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "nba":
        data = get_nba_dataset()
        points = data[:items]
        topk_ranking = [[i] for i in range(TK)]  # identity top-k
    else:
        data = get_jee_dataset(args.jee_csv, scale=bool(args.jee_scale))
        if len(data) < TK:
            raise ValueError(f"JEE dataset has only {len(data)} rows; need at least TK={TK}.")
        points = data[:items]
        topk_ranking = [[i] for i in range(TK)]

    points_np = np.array(points, dtype=float)
    ground_truth_topk = list(range(TK))

    
    # ---------- PHASE 1: Random Sampling ----------
    start = time.time()
    best_w, best_err = random_sampling_best(points_np, ground_truth_topk, samples=args.samples, seed=len(points)//len(topk_ranking), verbose=bool(args.verbose))
    t_rs = time.time() - start
    print(f"\n{'='*70}\nPHASE 1: Random Sampling\n{'='*70}")
    print(f"Best sampling error (Kendall-style): {best_err} (time {t_rs:.3f}s)")
    print(f"Best weights (L1-normalized, positive): {best_w}")


    # ---------- PHASE 2: Greedy Refinement ----------
    print(f"\n{'='*70}\nPHASE 2: Greedy Refinement (adjacent swaps + realizability)\n{'='*70}")
    start = time.time()
    refined_w, refined_err = greedy_refine_weights(points_np, best_w, ground_truth_topk, verbose=bool(args.greedy_verbose))
    t_gr = time.time() - start

    # Diagnostics for refined_w
    l2r = np.linalg.norm(refined_w) or 1.0
    alpha_unit_ref = (refined_w / l2r).tolist()
    summary_ref = lppp_diagnostics_counts(points, topk_ranking, alpha_unit_ref, strict=False, epsilon=1e-4)
    print(f"total time for LS-INV with {items} and {TK} is {t_gr}")
    print("summary", summary_ref)
   