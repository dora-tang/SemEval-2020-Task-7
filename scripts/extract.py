import re
import sys
# import ast
import argparse
import pandas as pd
# python scripts/extract.py -f exp1/*/log.log -csv task1.csv

parser = argparse.ArgumentParser(description='')
parser.add_argument('-f', type=str, nargs='*', default="", help='', required=True)
parser.add_argument('-csv', type=str, default="", help='bulk write to csv')
args = parser.parse_args()
col_base = ["model", "transformer", "finetune", "train_extra", "feature", "seed", ]
cols = col_base + ["val_rmse", "val_rmse10", "val_rmse20", "val_rmse30", "val_rmse40",
                   "test_rmse", "test_rmse10", "test_rmse20", "test_rmse30", "test_rmse40", 'path']
d = {k: "" for k in cols}
template = "{:<15}\t"*len(cols)
print(template.format(*cols))

df_data = {k: [] for k in cols}
# for file in sys.argv[1:]:
for file in args.f:
    d['path'] = file
    with open(file, 'r') as f:
        raw = f.read()
    try:
        raw = raw.replace('\n', '@@@@').replace('\t', '')
        config = re.findall(
            r"Arguments: ({.*?})", raw)[0].replace('@@@@', '').replace("true", "True").replace("false", "False")
        #config = ast.literal_eval(config)
        config = eval(config)
        keys = set(cols).intersection(set(config.keys()))
        for k in keys:
            d[k] = config[k]
        # val result
        val = re.findall("\[validation result(.*?)\]", raw)[0].replace('@@@@', '')
        pattern = 'RMSE: (.*).*RMSE@10: (.*).*RMSE@20: (.*).*RMSE@30: (.*).*RMSE@40: (.*)'
        val = re.findall(pattern, val)[0]
        d['val_rmse'], d['val_rmse10'],  d['val_rmse20'], d['val_rmse30'], d['val_rmse40'] = val
        # test result
        test = re.findall("\[test result(.*?)\]",
                          raw)[0].replace('@@@@', '')
        pattern = 'RMSE: (.*).*RMSE@10: (.*).*RMSE@20: (.*).*RMSE@30: (.*).*RMSE@40: (.*)'
        test = re.findall(pattern, test)[0]
        d['test_rmse'], d['test_rmse10'],  d['test_rmse20'], d['test_rmse30'], d['test_rmse40'] = test

        if d['model'] != 'transformer':
            d['transformer'] = ''
        v = [v for k, v in d.items()]
        print(template.format(*v))

        [df_data[k].append(v) for k, v in d.items()]
        # # d_list.append(d)
        # if df is None:
        #     df = pd.DataFrame(data=d, columns=cols)
        # else:
        #     df = pd.concat(df, pd.DataFrame(data=d, columns=cols))

        # print(result)
        #
        # "\t".join(cols) + "\n" +
        # l = "\t".join(x)
        # print(l)
        # d = {"val_rmse": val_rmse, "val_rmse10": val_rmse10, "val_rmse20": val_rmse20,
        # "val_rmse30": val_rmse30,  "val_rmse40": val_rmse40,  }
        #
        # test = re.findall("\[test result(.*?)\]", raw)[0].replace('@@@@', '\n').strip()
        # print(x)
    except IndexError as e:
        print(e)

df = pd.DataFrame(data=df_data,  columns=cols,).sort_values(by=col_base)
print(df)
if args.csv != "":
    df.to_csv(args.csv, sep=",", index=False)
