import glob, json, os, re
import numpy as np
import pandas as pd
from experiments.run_suite import CONFIGS

pat = re.compile(r"^(?P<cfg>.+)_s(?P<seed>\d+)_test_metrics\.json$")
rows = []
for path in glob.glob("results/tables/*_test_metrics.json"):
    m = pat.match(os.path.basename(path))
    if m:
        d = json.load(open(path))
        rows.append(dict(config=m["cfg"], seed=int(m["seed"]), **d))
if not rows:
    raise SystemExit("No seeded results found (expected results/tables/<cfg>_s<seed>_test_metrics.json)")

df = pd.DataFrame(rows).sort_values(["config", "seed"])
df.to_csv("results/tables/all_runs.csv", index=False)

order = [c for c in CONFIGS if c in set(df["config"])]
cols = [("accuracy", "Accuracy (%)"), ("precision_macro", "Precision (%)"),
        ("recall_macro", "Recall (%)"), ("f1_macro", "Macro-F1 (%)")]
summary = []
for cfg in order:
    g = df[df["config"] == cfg]
    row = {"Config": cfg, "n_seeds": len(g)}
    for key, label in cols:
        v = 100 * g[key].to_numpy()
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        row[label] = f"{v.mean():.2f} ± {sd:.2f}"
    row["Params"] = f"{int(g['n_params'].iloc[0]):,}" if "n_params" in g and g["n_params"].notna().all() else "n/a"
    summary.append(row)

out = pd.DataFrame(summary)
out.to_csv("results/tables/summary.csv", index=False)

md = ["| " + " | ".join(out.columns) + " |", "|" + "---|" * len(out.columns)]
md += ["| " + " | ".join(str(x) for x in r) + " |" for r in out.values]
open("results/tables/summary.md", "w").write("\n".join(md))

tex = ["\\begin{tabular}{" + "l" * len(out.columns) + "}", "\\hline",
       " & ".join(out.columns) + " \\\\ \\hline"]
tex += [" & ".join(str(x).replace("±", "$\\pm$").replace("_", "\\_") for x in r) + " \\\\" for r in out.values]
tex += ["\\hline", "\\end{tabular}"]
open("results/tables/summary.tex", "w").write("\n".join(tex))

print("\n".join(md))
print("\nSaved summary.csv / summary.md / summary.tex / all_runs.csv in results/tables/")
