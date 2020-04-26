from itertools import product
import sys

task = sys.argv[1]

# generate command for task1
if task == "1":
    prefix = 'python main.py -task task1 -bsz 32 -do_train -do_eval -save_model -track -tensorboard '

# generate command for task2
elif task == "2":
    prefix = 'python main.py -task task2 -bsz 16 -do_train -do_eval -save_model -track -tensorboard '


param_base = dict(
    feature = ['edit-context', 'edit-original'],
    train_extra = [0, 1],
    seed = [1],
)

# transformer_finetune
param1 = dict(
    model = ['transformer'],
    transformer = ['roberta-base', 'bert-base-uncased'],
    finetune = [1],
    lr = [2e-5, 5e-5],
    schedule = ['none', 'linear_schedule_with_warmup'],
    epoch = [3, 10],
)
param1.update(param_base)

# transformer_freeze
param2 = dict(
    model = ['transformer'],
    transformer = ['roberta-base', 'bert-base-uncased'],
    finetune = [0],
    lr = [ 1e-3, 3e-4],
    schedule = ['none',],
    epoch = [10],
)
param2.update(param_base)


#cbow_freeze_and_finetune
param3 = dict(
    model = ['cbow'],
    finetune = [0, 1],
    lr = [1e-3, 3e-4],
    schedule  = ["none"],
    epoch = [10],
)
param3.update(param_base)


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
        exp = "_".join([f"{k}_{v}" for k, v in d.items()])
        exp = exp.replace('model_cbow', 'cbow').replace('model_transformer_transformer_', '').replace('train_extra', 'extra')
        d['exp'] = exp
        cl = prefix + " ".join([f'-{k} {v}' for k, v in d.items()])
        print(cl)
        cl_list.append(cl)
    return cl_list


print('# transformer_finetue')
l1 = generate(param1)
print('# transformer_freeze')
l2 = generate(param2)
print('# cbow all')
l3 = generate(param3)
