# Prereqs
- Python 3.9+, Gurobi (licensed)
- Data: `./data/per.csv` (NBA) or `./data/jee.csv` (JEE).
# Install
```
pip install numpy pandas scikit-learn scipy gurobipy
```
# LP-PP & DP‑PP 

**LP‑PP (sample run (NBA dataset)):**
```bash
python main.py --dataset nba --items 1000 --tk 100   --lppp 1 --dppp 0   --outside_vs_mintop 1 --incremental_topk 1 --earlybreak 1
```
**DP‑PP (sample run (NBA dataset)):**
```bash
python main.py --dataset nba --items 1000 --tk 100 --lppp 0 --dppp 1
```
**LP‑PP (sample run (JEE dataset)):**
```bash
python main.py --dataset jee --jee_csv ./data/jee.csv --jee_scale 1   --items 20000 --tk 500 --lppp 1 --outside_vs_mintop 1 --incremental_topk 1 --earlybreak 1
```

## Core Flags (defaults in **bold**)
- `--dataset {nba,jee}` (default **nba**)
- `--jee_csv PATH` (default **./data/jee.csv**)
- `--items INT` (default **1000**), `--tk INT` (default **100**)
- **LP‑PP only:** `--outside_vs_mintop {0,1}` (**1**), `--incremental_topk {0,1}` (**1**), `--incremental_outside {0,1}` (**0**), `--earlybreak {0,1}` (**1**)
- Runner select: `--lppp {0,1}` (**1**), `--dppp {0,1}` (**0**)
- Misc: `--seed INT` (**10**), `--verbose {0,1}` (**0**)

## Outputs
- Learned `alpha` (unit‑L2), objective value, runtime.
- Diagnostics (PP, inversions, outsider violations, rank diffs).
- Logs in `app.log`.


# LS-INV

```
python ls_inv.py --dataset nba --items 1000 --tk 100 
```
## Core Flags:
In addition to --item and --tk (same as lp-pp and dp-pp), we have:
`--samples` (**1000**), which represents random samples for initial weights.

## Output
- **Phase 1:** best random sample error and weight vector  
- **Phase 2:** refined weights, total runtime, and diagnostic 
- Diagnostics (PP, inversions, outsider violations, rank diffs).

# HP-INV 
```
python hp_inv.py --dataset nba --items 1000 --tk 5 --tree_scope topk 
```
See the pratical consideration in section 5.2. Full-tree merge (no insertion phase):

```
python hp_inv.py --dataset nba --items 1000 --tk  --tree_scope all 
```

## Outputs
- Learned `alpha` (unit‑L2), runtime.
- Diagnostics (PP, inversions, outsider violations, rank diffs).
- Logs in `app.log`.

