import argparse, os, subprocess, sys

COMMON = ["--epochs", "30", "--batch_size", "64", "--lr", "1e-3", "--weight_decay", "0.05"]

CONFIGS = {
    # Tier 1: main comparison
    "bdh_main":   ["--model", "bdh"],
    "cnn":        ["--model", "cnn"],
    "vit":        ["--model", "vit"],
    # Tier 2: ablations + RGB vs 13-band (baselines too, so the claim is fair)
    "bdh_norope": ["--model", "bdh", "--no_rope"],
    "bdh_untied": ["--model", "bdh", "--no_tie_weights"],
    "bdh_rgb":    ["--model", "bdh", "--bands", "rgb"],
    "cnn_rgb":    ["--model", "cnn", "--bands", "rgb"],
    "vit_rgb":    ["--model", "vit", "--bands", "rgb"],
    # Tier 3: patch size and depth
    "bdh_p4":     ["--model", "bdh", "--patch_size", "4"],
    "bdh_p16":    ["--model", "bdh", "--patch_size", "16"],
    "bdh_L1":     ["--model", "bdh", "--n_layer", "1"],
    "bdh_L2":     ["--model", "bdh", "--n_layer", "2"],
}
TIERS = {
    1: ["bdh_main", "cnn", "vit"],
    2: ["bdh_norope", "bdh_untied", "bdh_rgb", "cnn_rgb", "vit_rgb"],
    3: ["bdh_p4", "bdh_p16", "bdh_L1", "bdh_L2"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=int, nargs="*", default=[1])
    p.add_argument("--only", nargs="*", default=None, help="run only these config names")
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    names = args.only if args.only else [n for t in args.tier for n in TIERS[t]]
    failed = []
    for seed in args.seeds:
        for cfg in names:
            run = f"{cfg}_s{seed}"
            if os.path.exists(f"results/tables/{run}_test_metrics.json"):
                print(f"[skip] {run} already done")
                continue
            train = [sys.executable, "-m", "experiments.train_bdh", "--name", run,
                     "--seed", str(seed), *COMMON, *CONFIGS[cfg]]
            evalc = [sys.executable, "-m", "experiments.evaluate", "--name", run]
            print("\n>>>", " ".join(train))
            if args.dry_run:
                continue
            if subprocess.run(train).returncode != 0 or subprocess.run(evalc).returncode != 0:
                failed.append(run)
    print("\nFailed runs:", failed if failed else "none")


if __name__ == "__main__":
    main()
