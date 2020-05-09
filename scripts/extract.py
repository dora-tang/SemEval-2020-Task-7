import re
import sys

# import ast
import argparse
import pandas as pd
import itertools

# python scripts/extract.py -f exp1/*/log.log -task task1 -csv task1.csv
# python scripts/extract.py -f exp1/*/log.log -task task1 -csv task1_best.csv -select_best 1
# python scripts/extract.py -f exp2/*/log.log -task task2 -csv task2.csv
# python scripts/extract.py -f exp2/*/log.log -task task2 -csv task2_best.csv -select_best 1

parser = argparse.ArgumentParser(description="")
parser.add_argument("-f", type=str, nargs="*", default="", help="", required=True)
parser.add_argument("-csv", type=str, default="", help="bulk write to csv")
parser.add_argument("-task", type=str, default="", help="")
parser.add_argument("-select_best", type=int, default=0, help="")
args = parser.parse_args()

col_base = [
    "model",
    "transformer",
    "finetune",
    "train_extra",
    "feature",
    "schedule",
    "lr",
    "epochs",
    "seed",
]
if args.task == "task1":
    col_task = [
        "val_rmse",
        "val_rmse10",
        "val_rmse20",
        "val_rmse30",
        "val_rmse40",
        "val_spearman",
        "val_pearson",
        "test_rmse",
        "test_rmse10",
        "test_rmse20",
        "test_rmse30",
        "test_rmse40",
        "test_spearman",
        "test_pearson",
        "path",
    ]
elif args.task == "task2":
    col_task = [
        "val_accuracy",
        "val_reward",
        "val_rmse",
        "test_accuracy",
        "test_reward",
        "test_rmse",
        "path",
    ]
else:
    raise ValueError

cols = col_base + col_task

d = {k: "" for k in cols}
template = "{:<15}\t" * len(cols)
print(template.format(*cols))

df_data = {k: [] for k in cols}
for file in args.f:
    d["path"] = file
    with open(file, "r") as f:
        raw = f.read()
    try:
        raw = raw.replace("\n", "@@@@").replace("\t", "")

        # config
        config = (
            re.findall(r"Arguments: ({.*?})", raw)[0]
            .replace("@@@@", "")
            .replace("true", "True")
            .replace("false", "False")
        )
        # config = ast.literal_eval(config)
        config = eval(config)
        keys = set(cols).intersection(set(config.keys()))
        for k in keys:
            d[k] = config[k]

        val = re.findall(r"validation result: ({.*?})", raw)[0].replace("@@@@", "")
        val = eval(val)
        val = {f"val_{k}": v for k, v in val.items()}
        keys = set(cols).intersection(set(val.keys()))
        for k in keys:
            d[k] = val[k]

        test = re.findall("test result: ({.*?})", raw)[0].replace("@@@@", "")
        test = eval(test)
        test = {f"test_{k}": v for k, v in test.items()}
        keys = set(cols).intersection(set(test.keys()))
        for k in keys:
            d[k] = test[k]

        if d["model"] != "transformer":
            d["transformer"] = ""
        v = [v for k, v in d.items()]
        print(template.format(*v))

        [df_data[k].append(v) for k, v in d.items()]

    except IndexError as e:
        print(e)

df = pd.DataFrame(data=df_data, columns=cols,).sort_values(by=col_base)

# filter row if schedule = none and epoch = 3
df = df[(df["schedule"] != "none") | (df["epochs"] != 3)]
# drop column
df = df.drop("seed", axis="columns", errors="ignore")


def modify_row(row):
    if row.model == "transformer":
        row.model = row.transformer
    row = row.drop("transformer")
    return row


df = df.apply(modify_row, axis=1)


if args.select_best != 0:
    # rows and cols
    model = ["cbow", "roberta-base", "bert-base-uncased"]
    finetune = [0, 1]
    train_extra = [0, 1]
    feature = ["edit-context", "edit-original"]
    rows = list(itertools.product(model, finetune, train_extra, feature))

    df_best = pd.DataFrame(columns=df.columns)
    for r in rows:
        model, finetune, train_extra, feature = r

        df0 = df[
            (df["model"] == model)
            & (df["finetune"] == finetune)
            & (df["train_extra"] == train_extra)
            & (df["feature"] == feature)
        ]

        if args.task == "task1":
            best_row = df0.iloc[df0["val_rmse"].values.argmin()]
        elif args.task == "task2":
            best_row = df0.iloc[df0["val_accuracy"].values.argmax()]

        df_best = df_best.append(best_row)

    col_base.remove("transformer")
    col_base.remove("seed")
    df_best.sort_values(by=col_base)

    print(df_best)
    if args.csv != "":
        df_best.to_csv(args.csv, sep=",", index=True)

else:
    print(df)
    if args.csv != "":
        df.to_csv(args.csv, sep=",", index=True)
