from utils.preprocessing import build_index, make_splits, compute_train_stats
from utils.seed import set_seed

set_seed(42)
df, class_to_idx = build_index("data/raw/EuroSAT_MS")
train_df, val_df, test_df = make_splits(df)

print(len(train_df), len(val_df), len(test_df))
print("Train class proportions:\n", train_df["class"].value_counts(normalize=True).round(3))
print("Test class proportions:\n", test_df["class"].value_counts(normalize=True).round(3))

mean, std = compute_train_stats(train_df)
print("mean:", mean.round(1))
print("std :", std.round(1))