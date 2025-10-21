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


# Configure logging to a file
file_handler = logging.FileHandler('app.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)


import math
import itertools
import gurobipy as gp
from itertools import combinations

def _normalize_tiers(topk_ranking):
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

    # scores under the provided alpha
    scores = [sum(alpha_unit[d]*points[i][d] for d in range(dims)) for i in range(n)]
    if args.verbose:
       print("scores", scores)

    tiers = _normalize_tiers(topk_ranking)
    ordered_topk = [i for tier in tiers for i in tier]
    topk_set = set(ordered_topk)
    outsiders = [i for i in range(n) if i not in topk_set]

    margin = 0.0 if strict else epsilon

    
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


  
    score_ranking = sorted(range(n), key=lambda i: scores[i], reverse=True)
    pos_score = {i: r for r, i in enumerate(score_ranking)}
    pos_given = {i: r for r, i in enumerate(ordered_topk)}  # minimal, expands ties as listed
    if args.verbose:
      print("pos_score", pos_score)
      print("pos_given", pos_given)
    # compute discrepancies only over top-k items
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


def lppp(points, topk_ranking, *,
         outside_vs_mintop: bool,
         incremental_outside: bool,
         incremental_topk: bool,
         earlybreak: bool,
         strict: bool = False,
         epsilon: float = 1e-4,
         max_rounds: int = 50,
         tol: float = 1e-12,
         outside_mode: str = "aggregate_exact"):
    """

      outside_vs_mintop=True
          Compare outsiders to z_min (score of the weakest top-k) instead of all top-k.
      incremental_topk=True
          Add inside-topk constraints incrementally (constraint generation).
      incremental_outside=True
          Add outsider-vs-topk constraints incrementally (ignored if outside_vs_mintop=True).

    Returns: (objective_value_L1, objective_value_rescaled_to_L2, alpha_unit_L2) or (None, None)
    """
    # ---- Model & solver knobs ----
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.Method = 2     
    model.Params.Presolve = 2
    model.Params.Crossover = 1
    model.Params.Seed = 1
    model.Params.Threads = 1
    model.Params.NumericFocus = 2

    n, dims = len(points), len(points[0])

   
    def _is_intlike(x):
        try:
            import numpy as _np
            return isinstance(x, (int,)) or isinstance(x, _np.integer)
        except Exception:
            return isinstance(x, int)

    if isinstance(topk_ranking, set):
        tiers = [list(int(i) for i in topk_ranking)]            # single tie tier
    elif hasattr(topk_ranking, "__iter__") and all(_is_intlike(x) for x in topk_ranking):
        tiers = [[int(i)] for i in topk_ranking]                # keep given order
    else:
        tiers = [[int(i) for i in tier] for tier in topk_ranking]

    ordered_topk = [i for tier in tiers for i in tier]
    topk_set = set(ordered_topk)
    outsiders = sorted(set(range(n)) - topk_set)

    tier_of = {}
    for ti, tier in enumerate(tiers):
        for idx in tier:
            tier_of[idx] = ti

    # ---- Variables ----
    alpha = model.addVars(range(dims), lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="alpha")
    model.addConstr(gp.quicksum(alpha[d] for d in range(dims)) == 1)

    def score_expr(i):
        return gp.quicksum(alpha[d] * points[i][d] for d in range(dims))

   
    error_vars = {}
    def get_err(i, j):
        if i > j: i, j = j, i
        if i not in error_vars:
            error_vars[i] = {}
        if j not in error_vars[i]:
            error_vars[i][j] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"e_{i}_{j}")
        return error_vars[i][j]

    obj = gp.LinExpr()

    
    z_min = None
    error_out_to_z = {}
    if outside_vs_mintop:
        z_min = model.addVar(lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="z_min")
        for tk in ordered_topk:
            model.addConstr(score_expr(tk) >= z_min)
        if strict:
            for o in outsiders:
                model.addConstr(score_expr(o) <= z_min)
        else:
            for o in outsiders:
                e = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"e_out_{o}")
                error_out_to_z[o] = e
                model.addConstr(e >= score_expr(o) - z_min + epsilon)
            obj += gp.quicksum(error_out_to_z.values())

    
    for tier in tiers:
        if len(tier) > 1:
            hub = tier[0]
            for j in tier[1:]:
                mi, mj = (hub, j) if hub < j else (j, hub)
                e = get_err(mi, mj)
                model.addConstr(e >= score_expr(hub) - score_expr(j))
                model.addConstr(e >= score_expr(j) - score_expr(hub))
                obj += e
 

    
    if not outside_vs_mintop:
        if incremental_outside:
            if strict:
                outside_mode = "pairwise_cg"
            if outside_mode not in ("pairwise_cg", "aggregate_exact"):
                raise ValueError("outside_mode must be 'pairwise_cg' or 'aggregate_exact'")
            if outside_mode == "aggregate_exact" and not strict:
                k = len(ordered_topk)
                y = {}
                e_out_sum = {}
                for o in outsiders:
                    e_out_sum[o] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"e_outsum_{o}")
                    so_plus = score_expr(o) + epsilon
                    rhs = k * so_plus
                    for t in ordered_topk:
                        y[(t, o)] = model.addVar(lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_{t}_{o}")
                        model.addConstr(y[(t, o)] <= score_expr(t))
                        model.addConstr(y[(t, o)] <= so_plus)
                        rhs -= y[(t, o)]
                    model.addConstr(e_out_sum[o] >= rhs)
                    obj += e_out_sum[o]
            # else: pure pairwise CG will be added in the loop
        else:
            # full add of all pairs
            for t in ordered_topk:
                for o in outsiders:
                    if strict:
                        model.addConstr(score_expr(t) - score_expr(o) >= 0.0)
                    else:
                        e = get_err(min(t, o), max(t, o))
                        model.addConstr(e >= score_expr(o) - score_expr(t) + epsilon)
                        obj += e

    

    # ---- Constraint generation loop  ----
    if incremental_topk:
        model.setObjective(obj, gp.GRB.MINIMIZE)
        margin_cross = 0.0 if strict else epsilon
        margin_out   = 0.0 if strict else epsilon

        added_cross = set()
        added_tie   = set()
        added_out   = set()

        # cross-tier: adjacent neighbours (+epsilon)
        for r in range(len(ordered_topk) - 1):
            i, j = ordered_topk[r], ordered_topk[r + 1]
            if tier_of[i] == tier_of[j]:
                continue
            mi, mj = (i, j) if i < j else (j, i)
            if strict:
                model.addConstr(score_expr(i) - score_expr(j) >= 0.0)
            else:
                e = get_err(mi, mj)
                model.addConstr(e >= score_expr(j) - score_expr(i) + epsilon)
                obj += e

        for i in error_vars:
            for j in error_vars[i]:
                if tier_of.get(i, -1) == tier_of.get(j, -1):
                    added_tie.add((i, j))
                else:
                    added_cross.add((i, j))

        for _ in range(max_rounds):
            model.optimize()
            if model.SolCount == 0:
                return (None, None)

            alpha_vals = [alpha[d].X for d in range(dims)]
            scores = [sum(alpha_vals[d] * points[i][d] for d in range(dims)) for i in range(n)]

            new_added = 0

            # (1) cross-tier: add violated earlier->later pairs
            for later_pos in range(len(ordered_topk)):
                t = ordered_topk[later_pos]
                for earlier_pos in range(later_pos):
                    c = ordered_topk[earlier_pos]
                    if tier_of[c] == tier_of[t]:
                        continue
                    if scores[t] - scores[c] + margin_cross > tol:
                        mi, mj = (c, t) if c < t else (t, c)
                        if (mi, mj) in added_cross:
                            continue
                        if strict:
                            model.addConstr(score_expr(c) - score_expr(t) >= 0.0)
                        else:
                            e = get_err(mi, mj)
                            model.addConstr(e >= score_expr(t) - score_expr(c) + margin_cross)
                            obj += e
                        added_cross.add((mi, mj))
                        new_added += 1

            # (2) ties within tier (no epsilon)
            for tier in tiers:
                if len(tier) <= 1:
                    continue
                for i, j in combinations(tier, 2):
                    mi, mj = (i, j) if i < j else (j, i)
                    if (mi, mj) in added_tie:
                        continue
                    if abs(scores[i] - scores[j]) > tol:
                        e = get_err(mi, mj)
                        model.addConstr(e >= score_expr(i) - score_expr(j))
                        model.addConstr(e >= score_expr(j) - score_expr(i))
                        obj += e
                        added_tie.add((mi, mj))
                        new_added += 1

            # (3) outsiders (only when not using z_min)
            if not outside_vs_mintop:
                if strict or outside_mode == "pairwise_cg":
                    for tk in ordered_topk:
                        s_tk = scores[tk]
                        for o in outsiders:
                            if scores[o] - s_tk + margin_out > tol:
                                mi, mj = (tk, o) if tk < o else (o, tk)
                                if (mi, mj) in added_out:
                                    continue
                                if strict:
                                    model.addConstr(score_expr(tk) - score_expr(o) >= 0.0)
                                else:
                                    e = get_err(mi, mj)
                                    model.addConstr(e >= score_expr(o) - score_expr(tk) + margin_out)
                                    obj += e
                                added_out.add((mi, mj))
                                new_added += 1
                

            if new_added == 0:
                print(f"number of rounds {_} ")
                break
            if earlybreak:
               print(f"number of rounds {_} ")
               break

            model.setObjective(obj, gp.GRB.MINIMIZE)
    else:
        curr_proc = []
        for tkr in topk_ranking:
            for c in curr_proc:
                for t in tkr:
                    e = get_err(min(c, t), max(c, t))
                    model.addConstr(e >= score_expr(t) - score_expr(c) + epsilon)
                    obj += e

            
            for t in tkr:
                curr_proc.append(t)

            
            if len(tkr) > 1:
                
                for pairs in combinations(tkr, 2):
                    
                    i, j = pairs
                    mi, mj = min(i, j), max(i, j)
                    e = get_err(mi, mj)
                    model.addConstr(e >= score_expr(i) - score_expr(j))
                    model.addConstr(e >= score_expr(j) - score_expr(i))
                    obj += e


    
    model.setObjective(obj, gp.GRB.MINIMIZE)
    model.optimize()
    if model.SolCount == 0:
        return (None, None)

    alpha_vals = [alpha[d].X for d in range(dims)]
    l1 = sum(abs(v) for v in alpha_vals)
    assert abs(l1 - 1) <= 1e-4

    l2 = math.sqrt(sum(v*v for v in alpha_vals)) or 1.0
    alpha_unit = [v / l2 for v in alpha_vals]
    print("opt", model.ObjVal)
    return model.ObjVal, (model.ObjVal / l2), alpha_unit


def get_nba_dataset():
  df = pd.read_csv('./data/per.csv', skiprows=1, header=0).dropna()
  df = df[df.columns[1:9]]
  scaler = MinMaxScaler()
  df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
  return df_scaled.values.tolist()


def find_hyperplane_through_points(points):
    
    d = points.shape[1]
    
    A = np.hstack((points, np.ones((d, 1))))

    
    U, S, Vt = np.linalg.svd(A)
    
    null_space_vector = Vt[-1, :]

    w = null_space_vector[:-1]
    b = null_space_vector[-1]


    if np.all(w<0):
      w *= -1
      b *= -1
    return w, b

def get_intersection_point(w, b):
  lambda_val = abs(-b)/sum(map(lambda x:x*x, w))
  point = lambda_val*w
  norm_point = point/np.linalg.norm(point)
  return norm_point

def farthest_point(w, b):

  
  model = gp.Model()
  dims = len(w)

  alpha = model.addVars(list(range(dims)), lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="alpha")

  # Equation of the ball, all inclusive
  model.addConstr(gp.quicksum(alpha[i]*alpha[i] for i in range(dims)) <= 1)

  # Equation of plane
  model.addConstr(gp.quicksum(w[i]*alpha[i] for i in range(dims)) + b == 0)

  model.setObjective(gp.quicksum(alpha[i]*alpha[i] for i in range(dims)), gp.GRB.MINIMIZE)
  model.optimize()

  # Check if Optimization Successful!
  if model.SolCount > 0:
    alpha_vals = [alpha[i].X for i in range(dims)]

    
    l1_norm = sum(alpha_vals[i]* w[i] for i in range(dims)) + b
    assert abs(l1_norm) <= 0.0001

    
    optimal_value = model.ObjVal
    return math.sqrt(optimal_value), alpha_vals
  else:
   
    return None, None

def non_convex_opt(points, topk_ranking, strict=False, epsilon = 0.0001):

  
  model = gp.Model()

  n = len(points)
  dims = len(points[0])

  # Get items in topk
  topk = set()
  for tkr in topk_ranking:
    for item in tkr:
      topk.add(item)

  # Get not in top k
  not_in_topk = set(range(n)) - topk

  error_vars = dict()
  for pairs in combinations(list(range(n)), 2):
    i, j = min(pairs), max(pairs)

    if i not in error_vars:
      error_vars[i] = dict()

    # Do not allow negative errors
    error_vars[i][j] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"error_{i}_{j}", lb=0.0)

  alpha = model.addVars(list(range(dims)), lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="alpha")

  # Add the constraint that sum of alpha squares is 1
  model.addConstr(gp.quicksum(alpha[i]*alpha[i] for i in range(dims)) == 1)

  curr_proc = []
  for tkr in topk_ranking:
    
    for c in curr_proc:
      for t in tkr:
        
        model.addConstr(error_vars[min(c, t)][max(c, t)] >= gp.quicksum(alpha[_i]*(points[t][_i] - points[c][_i]) for _i in range(dims)) + epsilon)


    # Add items in topk level to current list
    for t in tkr:
      curr_proc.append(t)

    # If tkr is greater than one then all elements in same level should have same score, else penalise
    if len(tkr) > 1:
      # Do pair wise error computation
      for pairs in combinations(tkr, 2):
        # Add error to the error function
        i, j = pairs
        mi, mj = min(i, j), max(i, j)
        model.addConstr(error_vars[mi][mj] >= gp.quicksum(alpha[_i]*(points[i][_i] - points[j][_i]) for _i in range(dims)))
        model.addConstr(error_vars[mi][mj] >= gp.quicksum(alpha[_i]*(points[j][_i] - points[i][_i]) for _i in range(dims)))


  for tk in itertools.chain.from_iterable(topk_ranking):
    for ntk in not_in_topk:
      if strict:
        model.addConstr(gp.quicksum(alpha[_i]*(points[tk][_i] - points[ntk][_i]) for _i in range(dims)) >= 0)
      else:
        # Consider these as violations if tk is not better than ntk by atleast epsilon
        model.addConstr(error_vars[min(tk, ntk), max(tk, ntk)] >= gp.quicksum(alpha[_i]*(points[ntk][_i] - points[tk][_i]) for _i in range(dims)) + epsilon)
  model.setObjective(gp.quicksum(error_vars[i][j] for i in range(n) for j in range(i + 1 ,n)), gp.GRB.MINIMIZE)
  model.optimize()

  # Check if Optimization Successful!
  if model.SolCount > 0:
    alpha_vals = [alpha[i].X for i in range(dims)]

    # Step 1 : Obtain the alpha function

    # Check that alpha_vars l1 norm adds to 1
    l2_norm = sum(alpha_vals[i]*alpha_vals[i] for i in range(dims))
    assert abs(l2_norm-1) <= 0.0001

    # Get optimization score value
    optimal_value = model.ObjVal
    return optimal_value, alpha_vals
  else:
    # Optimization score does not exist
    return None, None

def dlppp(points, topk_ranking, strict=False, epsilon = 0.0001):
  n = len(points)
  dims = len(points[0])

  # Generate empty parameters
  min_error = math.inf
  min_alpha = []

  # Create model
  model = gp.Model()

  # Get items in topk
  topk = set()
  for tkr in topk_ranking:
    for item in tkr:
      topk.add(item)

  # Get not in top k
  not_in_topk = set(range(n)) - topk

  error_vars = dict()
  for pairs in combinations(list(range(n)), 2):
    i, j = min(pairs), max(pairs)

    if i not in error_vars:
      error_vars[i] = dict()

    # Do not allow negative errors
    error_vars[i][j] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"error_{i}_{j}", lb=0.0)

  alpha = model.addVars(list(range(dims)), lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="alpha")

  curr_proc = []
  for tkr in topk_ranking:
    # Add error computations for points which are ranked better than the current level of tkr
    for c in curr_proc:
      for t in tkr:
        model.addConstr(error_vars[min(c, t)][max(c, t)] >= gp.quicksum(alpha[_i]*(points[t][_i] - points[c][_i]) for _i in range(dims)) + epsilon)


    # Add items in topk level to current list
    for t in tkr:
      curr_proc.append(t)

    # If tkr is greater than one then all elements in same level should have same score, else penalise
    if len(tkr) > 1:
      for pairs in combinations(tkr, 2):
        i, j = pairs
        mi, mj = min(i, j), max(i, j)
        model.addConstr(error_vars[mi][mj] >= gp.quicksum(alpha[_i]*(points[i][_i] - points[j][_i]) for _i in range(dims)))
        model.addConstr(error_vars[mi][mj] >= gp.quicksum(alpha[_i]*(points[j][_i] - points[i][_i]) for _i in range(dims)))


  for tk in itertools.chain.from_iterable(topk_ranking):
    for ntk in not_in_topk:
      if strict:
        model.addConstr(gp.quicksum(alpha[_i]*(points[tk][_i] - points[ntk][_i]) for _i in range(dims)) >= 0)
      else:
        # Consider these as violations if tk is not better than ntk by atleast epsilon
        try:
          model.addConstr(error_vars[min(tk, ntk)][max(tk, ntk)] >= gp.quicksum(alpha[_i]*(points[ntk][_i] - points[tk][_i]) for _i in range(dims)) + epsilon)
        except:
          import pdb
          pdb.set_trace()
  model.setObjective(gp.quicksum(error_vars[i][j] for i in range(n) for j in range(i + 1 ,n)), gp.GRB.MINIMIZE)

  plane_constr = None

  for dim in range(dims):
    if plane_constr is not None:
      model.remove(plane_constr)

    
    plane_constr = model.addConstr(gp.quicksum(alpha[i] if not(i == dim) else -(dims-1-math.sqrt(dims))*alpha[i] for i in range(dims)) == 1)

    model.optimize()

    # Check if Optimization Successful!
    if model.SolCount > 0:
      alpha_vals = [alpha[i].X for i in range(dims)]

      
      l1_norm = sum(alpha_vals[i] if not(i == dim) else -(dims-1-math.sqrt(dims))*alpha_vals[i] for i in range(dims))
      assert abs(l1_norm-1) <= 0.0001

      
      l2_norm = math.sqrt(sum(math.pow(alpha_vals[i], 2) for i in range(dims)))
      alpha_vals = [alpha_vals[i]/l2_norm for i in range(dims)]

      
      optimal_value = model.ObjVal/l2_norm
      if optimal_value < min_error:
        min_error = optimal_value
        min_alpha = alpha_vals
    else:
      
      return None, None
  
  l2 = math.sqrt(sum(v*v for v in alpha_vals)) or 1.0
  alpha_unit = [v / l2 for v in alpha_vals]
  return model.ObjVal, (model.ObjVal / l2), alpha_unit
  

def ilp_inv(points, topk_ranking, strict=False, epsilon = 0.0001):

  n = len(points)
  dims = len(points[0])

  # Create model
  model = gp.Model()

  alpha = model.addVars(list(range(dims)), lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="alpha")

  
  model.addConstr(gp.quicksum(alpha[i] for i in range(dims)) == 1)

  
  large_const = 2*dims

  # Get items in topk
  topk = set()
  for tkr in topk_ranking:
    for item in tkr:
      topk.add(item)

  # Get not in top k
  not_in_topk = set(range(n)) - topk

  error_vars = dict()
  for ti in topk:
    for ntj in range(n):
      if ti == ntj:
        continue
      i, j = min(ti, ntj), max(ti, ntj)

      if i not in error_vars:
        error_vars[i] = dict()

      
      error_vars[i][j] = model.addVar(vtype=gp.GRB.BINARY, name=f"error_{i}_{j}", lb=0)

  curr_proc = []
  for tkr in topk_ranking:
    for c in curr_proc:
      for t in tkr:
        
        model.addConstr(large_const*error_vars[min(c, t)][max(c, t)] >= gp.quicksum(alpha[_i]*(points[t][_i] - points[c][_i]) for _i in range(dims)) + epsilon)
    # Add items in topk level to current list
    for t in tkr:
      curr_proc.append(t)

    # If tkr is greater than one then all elements in same level should have same score, else penalise
    if len(tkr) > 1:
      # Do pair wise error computation
      for pairs in combinations(tkr, 2):
        # Add error to the error function
        i, j = pairs
        mi, mj = min(i, j), max(i, j)
        
        model.addConstr(large_const*error_vars[mi][mj] >= gp.quicksum(alpha[_i]*(points[i][_i] - points[j][_i]) for _i in range(dims)))
        model.addConstr(large_const*error_vars[mi][mj] >= gp.quicksum(alpha[_i]*(points[j][_i] - points[i][_i]) for _i in range(dims)))



  for tk in itertools.chain.from_iterable(topk_ranking):
    for ntk in not_in_topk:
      if strict:
        model.addConstr(gp.quicksum(alpha[_i]*(points[tk][_i] - points[ntk][_i]) for _i in range(dims)) >= 0)
      else:
        
        model.addConstr(large_const*error_vars[min(tk, ntk), max(tk, ntk)] >= gp.quicksum(alpha[_i]*(points[ntk][_i] - points[tk][_i]) for _i in range(dims)) + epsilon)
  model.setObjective(gp.quicksum(error_vars[i][j] for i in range(n) for j in range(i + 1 ,n)), gp.GRB.MINIMIZE)
  model.optimize()

 
  if model.SolCount > 0:
    alpha_vals = [alpha[i].X for i in range(dims)]

    
    l1_norm = sum(alpha_vals[i] for i in range(dims))
    assert abs(l1_norm-1) <= 0.0001

    # Get optimization score value
    optimal_value = model.ObjVal
    return optimal_value, alpha_vals
  else:
    # Optimization score does not exist
    return None, None

def partial_shuffle(seq, tk, seed=None):
    """
    Permute only a fraction of positions in `seq` in-place.
    Guarantees all selected positions change (via 1-step rotation).
    """
    rng = random.Random(seed)
    n = len(seq)

    # choose k distinct positions
    k = int(1.5 * tk)
    idxs = rng.sample(range(min(k, n)), tk)
    idxs.sort()  # order doesn’t matter; sort for reproducibility
    # rotate the chosen values by 1 so no element stays in place
    vals = [seq[i] for i in idxs]
    vals_rot = vals[-1:] + vals[:-1]
    for i, v in zip(idxs, vals_rot):
        seq[i] = v
    return seq

def get_jee_dataset(csv_path: str, drop_cols=None, scale=False):
    """
    Reads a JEE CSV and returns a numeric feature matrix (list of lists),
    keeping marks and adding encoded category/GENDER/is_pwd.

    Encodings:
      - GENDER: M -> 0, F -> 1
      - category: GEN -> 0, OBC-NCL -> 1, SC -> 2, ST -> 3
      - is_pwd: True/Yes/1 -> 1, False/No/0 -> 0
    """
    if drop_cols is None:
        drop_cols = ["RANK", "BLOCK_ID", "AIR", "REGST_NO", "mark"]

    # Read with stable inference
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["__orig_idx__"] = np.arange(len(df))

    # --- Clean & encode categorical ---
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
            if pd.isna(x): return np.nan
            if isinstance(x, str):
                s = x.strip().lower()
                if s in {"1","true","t","yes","y"}: return 1.0
                if s in {"0","false","f","no","n"}: return 0.0
            try:
                return 1.0 if bool(int(x)) else 0.0
            except Exception:
                return 1.0 if bool(x) else 0.0
        df["is_pwd"] = df["is_pwd"].apply(_to_bin)

   
    for col in ["math", "phys", "chem", "mark"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

   
    to_drop = [c for c in drop_cols if c in df.columns and c != "__orig_idx__"]
    df = df.drop(columns=to_drop, errors="ignore")

    # Keep only numeric columns
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.empty:
        raise ValueError("No numeric columns found after encoding and dropping IDs.")

    
    encoded_cols = [c for c in ["category", "GENDER"] if c in num_df.columns]
    cont_cols = [c for c in ["math", "phys", "chem"] if c in num_df.columns]
    if scale and cont_cols:
        scaler = MinMaxScaler()
        num_df[cont_cols] = scaler.fit_transform(num_df[cont_cols].values)

   
    num_df = num_df.sort_values("__orig_idx__")

    
    ordered = [c for c in ["category", "GENDER",  "math", "phys", "chem"] if c in num_df.columns]
    out = num_df[ordered].to_numpy().tolist()

   
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["nba", "jee"], default="nba",
                        help="Choose dataset: nba (default) or jee")
    parser.add_argument("--jee_csv", type=str, default="./data/jee.csv",
                        help="Path to JEE CSV when --dataset jee")
    parser.add_argument("--jee_scale", type=int, default=1,
                        help="If 1, apply MinMax scaling to JEE features (default 0)")
    parser.add_argument("--items", type=int, default=1000)
    parser.add_argument("--tk", type=int, default=100)
    parser.add_argument("--outside_vs_mintop", type=int, default=1)
    parser.add_argument("--incremental_outside", type=int, default=0)
    parser.add_argument("--incremental_topk", type=int, default=1)
    parser.add_argument("--earlybreak", type=int, default=1)
    parser.add_argument("--randomsuffle", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--lppp", type=int, default=1)
    parser.add_argument("--dppp", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    TK = args.tk
    items = args.items
    random.seed(args.seed)

    if args.dataset == "nba":
        data = get_nba_dataset()
        points = data[:items]
        topk_ranking = [[i] for i in range(TK)]

    else:
        
        data = get_jee_dataset(args.jee_csv, scale=bool(args.jee_scale))
        if len(data) < TK:
            raise ValueError(f"JEE dataset has only {len(data)} rows; need at least TK={TK}.")
        points = data[:items]
        
        
        topk_ranking = [[i] for i in range(TK)]
    if args.verbose:
      print(points)
      print(topk_ranking)
    
    if args.lppp:
        start = time.time()
        output_linear = lppp(
            points,
            topk_ranking,
            outside_vs_mintop=bool(args.outside_vs_mintop),
            incremental_outside=bool(args.incremental_outside),
            incremental_topk=bool(args.incremental_topk),
            earlybreak=bool(args.earlybreak),
            strict=False,
            epsilon=1e-4
        )
        end = time.time()
        time_diff_lppp = end - start
        print(output_linear[2])
        print(f"total time for lppp with {items} and {TK} is {time_diff_lppp} and opt is {output_linear[0]}")
        logging.info(f"LP_unified {items} {TK} {output_linear},  TIME : {time_diff_lppp}")
        summary = lppp_diagnostics_counts(points, topk_ranking, output_linear[2], strict=False, epsilon=1e-4)
        print("summary", summary)
    if args.dppp:
        start = time.time()
        output_linear_dlppp = dlppp(points, topk_ranking, strict=False, epsilon = 0.0001)
        end = time.time()
        time_diff_lppp = end - start
        print(output_linear_dlppp)
        print(f"total time for dlppp with {items} and {TK} is {time_diff_lppp} and opt is {output_linear_dlppp[0]}")
        logging.info(f"dlLP_unified {items} {TK} {output_linear_dlppp},  TIME : {time_diff_lppp}")
        summary = lppp_diagnostics_counts(points, topk_ranking, output_linear_dlppp[2], strict=False, epsilon=1e-4)
        print("summary", summary)
    

