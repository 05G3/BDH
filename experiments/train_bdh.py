import argparse, json, math, os, time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.eurosat_dataset import EuroSATMS
from models.classifier import BDHClassifier
from utils.seed import set_seed
from utils.metrics import run_eval


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="bdh_main")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=0, help="smoke test: limit batches per epoch")
    # model
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=8)   # mlp_internal_dim_multiplier
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--causal", action="store_true")     # ablation only; leave off for images
    p.add_argument("--no_rope", action="store_true")    # ablation: use learned pos-embed instead
    p.add_argument("--no_tie_weights", action="store_true")  # ablation: untie layer weights
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = EuroSATMS("train", train=True)
    val_ds = EuroSATMS("val", train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model_cfg = dict(n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
                 mlp_internal_dim_multiplier=args.mlp_mult, dropout=args.dropout,
                 patch_size=args.patch_size, causal=args.causal,
                 use_rope=not args.no_rope, tie_weights=not args.no_tie_weights)
    model = BDHClassifier(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader) if not args.max_batches else args.max_batches
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    ckpt_path = f"results/checkpoints/{args.name}_best.pt"
    log_path = f"results/logs/{args.name}_history.csv"

    history, best_val_acc, bad_epochs = [], 0.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, run_correct, run_n = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for i, (x, y) in enumerate(pbar):
            if args.max_batches and i >= args.max_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            run_loss += loss.item() * y.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            run_n += y.size(0)
            pbar.set_postfix(loss=f"{run_loss / run_n:.3f}", acc=f"{run_correct / run_n:.3f}")

        train_loss, train_acc = run_loss / run_n, run_correct / run_n
        val_loss, val_acc, _, _ = run_eval(model, val_loader, device, criterion)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | lr {lr_now:.2e} | "
              f"{time.time() - t0:.0f}s")

        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, lr=lr_now))
        pd.DataFrame(history).to_csv(log_path, index=False)

        if val_acc > best_val_acc:
            best_val_acc, bad_epochs = val_acc, 0
            torch.save({"model_state": model.state_dict(), "config": model_cfg,
                        "epoch": epoch, "val_acc": val_acc}, ckpt_path)
            print(f"  -> saved best (val acc {val_acc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Done. Best val acc: {best_val_acc:.4f}  ({ckpt_path})")


if __name__ == "__main__":
    main()