import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.eurosat_dataset import EuroSATMS
from models.classifier import BDHClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="bdh_main")
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(f"results/checkpoints/{args.name}_best.pt", map_location=device)
    model = BDHClassifier(**ckpt["config"]).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    model.bdh.record_stats = True

    ds = EuroSATMS(args.split, bands=ckpt.get("bands"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    agg, n = None, 0
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device))
            bs = x.size(0)
            if agg is None:
                agg = [{k: 0.0 for k in s} for s in model.bdh.stats]
            for lvl, s in enumerate(model.bdh.stats):
                for k, v in s.items():
                    agg[lvl][k] += v * bs
            n += bs

    rows = []
    for lvl, s in enumerate(agg, start=1):
        rows.append({"level": lvl, **{k: 100 * v / n for k, v in s.items()}})
    df = pd.DataFrame(rows).round(2)
    df.columns = ["level", "x_sparse_%zeros", "y_sparse_%zeros", "xy_sparse_%zeros"]
    print(f"\nActivation sparsity ({args.name}, {args.split} split, {n} images)")
    print("(percent of exactly-zero entries after ReLU)\n")
    print(df.to_string(index=False))
    df.to_csv(f"results/tables/{args.name}_sparsity.csv", index=False)


if __name__ == "__main__":
    main()
