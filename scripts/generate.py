# generate command for task1

from itertools import product

prefix = 'python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track '


param_base = {
    "feature": ['edit-context', 'edit-original'],
    "finetune": [0, 1],
    "train_extra": [0, 1],
    "seed": [1],
}

param1 = {"model": ['cbow'], }
param1.update(param_base)

param2 = {
    "model": ['transformer'],
    "transformer": ['roberta-base', 'bert-base-uncased'],
}
param2.update(param_base)


def generate(param):
    combinations = list(product(*[v for k, v in param.items()]))
    cl_list = []
    for v_list in combinations:
        model = v_list[0]
        transformer = v_list[1]
        d = {k: v for k, v in zip(param.keys(), v_list)}
        if d['model'] == 'transformer':
            model = d['transformer']
        else:
            model = d['model']
        exp = f"{model}_{d['feature']}_fintune{d['finetune']}_extra{d['train_extra']}_seed{d['seed']}"
        d['exp'] = exp
        cl = prefix + " ".join([f'-{k} {v}' for k, v in d.items()])
        # d_list.append(d)
        print(cl)
        cl_list.append(cl)
    return cl_list


print('# cbow')
l1 = generate(param1)
print('# transformer')
l2 = generate(param2)
