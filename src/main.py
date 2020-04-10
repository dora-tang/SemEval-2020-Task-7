import sys
import argparse
from models import *
# from util import *


def parse_args(cl_args):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-task', type=str, default="task1", choices=['task1', 'task2', ], help='')
    parser.add_argument('-bsz', type=int, default=32, help='param: batch size')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-epochs', type=int, default=10, help='max epochs')
    # prevent gradient norm from being too large
    parser.add_argument('-grad_norm', type=float, default=5, help='gradient norm clipping')

    parser.add_argument('-model', type=str, default='cbow',
                        choices=['cbow', 'transformer', ], help='')
    # chocies: 'bert-base-uncased', 'bert-large-uncased','roberta-base', 'roberta-large', 'roberta-large-mnli'
    parser.add_argument('-transformer', type=str, default='bert-base-uncased', help='')
    # parser.add_argument('-finetune', action='store_true', help='')
    parser.add_argument('-finetune', type=int, default=0, choices=[0, 1], help='')
    #parser.add_argument('-train_extra', action='store_true', help='')
    parser.add_argument('-feature',  type=str, default='edit-context',
                        choices=['edit-context', 'edit-original', ], help='')

    parser.add_argument('-train_extra', type=int, default=0, choices=[0, 1],
                        help='use funlines dataset as additional dataset if 1')

    parser.add_argument('-exp',  type=str, default='exp', help='experiment directory')
    # choose pipeline
    parser.add_argument('-do_train', action='store_true', help='')
    parser.add_argument('-do_eval', action='store_true', help='')
    parser.add_argument('-do_predict', action='store_true', help='')
    # additional options
    parser.add_argument('-track', action='store_true', help='track training time')
    parser.add_argument('-save_model', action='store_true', help='save model state')
    parser.add_argument('-tensorboard', action='store_true',
                        help='use tensorboard for logging metrics')
    # paths
    parser.add_argument('-train_path', type=str, default="", help='')
    parser.add_argument('-train_extra_path', type=str, default="", help='')
    parser.add_argument('-val_path', type=str, default="", help='')
    parser.add_argument('-test_without_label_path', type=str, default="", help='')
    parser.add_argument('-test_with_label_path', type=str, default="", help='')
    # parser.add_argument('-cache', type=str, default='cache/transformers',
    #                     help='cache directory for pretrained transformers')
    # parser.add_argument('-model_path', action='store_true', help='')

    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-schedule', type=str, default="none",
                        choices=["none", "reduce_on_plateau", "linear_schedule_with_warmup"], help='')
    args = parser.parse_args(cl_args)
    return args


args = parse_args(sys.argv[1:])

if args.task == 'task1':
    # task 1
    from base1 import *
    #from base1 import calc_rmse as full_eval
    # from base1 import evaluate as full_eval
    from code.score_task_1 import score_task_1 as score_task
    default_data_dir = '../data/task-1'
    exp_dir = os.path.join('../exp1', args.exp)
    out_path = os.path.join(exp_dir, 'task-1-output.csv')

elif args.task == 'task2':
    from base2 import *
    # from base2 import evaluate as full_eval
    from code.score_task_2 import score_task_2 as score_task
    default_data_dir = '../data/task-2'
    exp_dir = os.path.join('../exp2', args.exp)
    out_path = os.path.join(exp_dir, 'task-2-output.csv')

args.exp_dir = exp_dir
args = args_default_path(args, default_data_dir)
config_path = os.path.join(exp_dir, 'params.json')
model_path = os.path.join(exp_dir, 'model_state.th')
log_path = os.path.join(exp_dir, 'log.log')
seed_all(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_log()


if args.do_train:
    # clear experiment directory
    if os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    log.getLogger().addHandler(log.FileHandler(log_path, mode='w'))
    log.info('---- Phase: train -----')
    with open(config_path, 'w') as conf:
        args_dict = copy.deepcopy(vars(args))
        args_exclude = ['exp', 'track', 'save_model', 'tensorboard', 'do_eval', 'do_train', 'do_predict',
                        'train_path', 'train_extra_path', 'val_path', 'test_with_label_path', 'test_without_label_path', 'exp_dir', ]
        for k in args_exclude:
            del args_dict[k]

        json.dump(args_dict, conf, indent=2)
else:
    # assert experiment directory exists
    assert os.path.isdir(exp_dir)
    assert os.path.exists(config_path)
    assert os.path.exists(model_path)

    log.getLogger().addHandler(log.FileHandler(log_path, mode='a'))
    log.info('\n\n###################################')
    with open(config_path, 'r') as conf:
        args_dict = json.load(conf)
        # log.info(args_dict)
        for k, v in args_dict.items():
            vars(args)[k] = v
            # log.info(args)
            # log.info(k)
            # log.info(v)

args_json = json.dumps(vars(args), indent=2)

log.info(f'Arguments: {args_json}')
log.info(f'Device: {device}')


''' Preprocessing (Model Specific) '''

if args.model == 'transformer':
    # model_name = args.transformer
    tokenizer, transformer = init_transformers(args.transformer)
    TEXT = data.Field(tokenize=lambda x: tokenize_and_cut(x, tokenizer=tokenizer),
                      batch_first=True,
                      use_vocab=False,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      # init_token = init_token_idx,
                      # eos_token = eos_token_idx,
                      pad_token=tokenizer.pad_token_id,
                      unk_token=tokenizer.unk_token_id,)
    mask_token = tokenizer.mask_token
elif args.model == 'cbow':
    TEXT = data.Field(tokenize='spacy', sequential=True, batch_first=True, use_vocab=True)
    mask_token = None

task = get_task(args, get_dataset, TEXT, mask_token)

''' Define Model '''

if args.model == 'cbow':
    TEXT, embedding = load_word_embedding(TEXT, task['train_data'])

    embedding.weight.requires_grad = bool(args.finetune)

    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model = CBOW(embedding, pad_idx=PAD_IDX, feature=args.feature, mode=args.task, )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


elif args.model == 'transformer':

    model = PretrainedTransformer(transformer,
                                  finetune=bool(args.finetune),
                                  feature=args.feature,
                                  mode=args.task,
                                  pad_token_id=tokenizer.pad_token_id,
                                  sep_token_id=tokenizer.sep_token_id,)
    optimizer = huggingface_AdamW(model.parameters())

#optimizer = optim.Adam(model.parameters(), lr=args.lr)
# adam_epsilon = 1e-8
# optimizer = AdamW(model.parameters(), lr=args.lr, eps=adam_epsilon)

#optimizer = optim.Adam(model.parameters(), lr=args.lr)
#optimizer = huggingface_AdamW(model.parameters(), lr=args.lr, )


''' Training '''

if args.do_train:
    log.info(f'The model has {count_parameters(model)} trainable parameters')
    # log.info(model)
    best_model = train_loop(args, model, optimizer,
                            train_data=task['train_data'], val_data=task['val_data'],)

    #log.info('\nTest best epoch on validation data')
    log.info('\n[validation result')
    evaluate(args, best_model, task['val_data'])
    log.info(']')

    # save model
    if args.save_model:
        torch.save(best_model.state_dict(), model_path)
        log.info(f"\nSave best model state to {model_path}")
    else:
        log.info('\nNOT save model state.')

''' Evaluation '''

if args.do_eval:
    log.info('\n---- Phase: eval -----')
    if not args.do_train:
        model.load_state_dict(torch.load(model_path))
    else:
        model = best_model
    log.info(f'Evaluate: {args.test_with_label_path}')

    # eval
    log.info('\n[test result')
    evaluate(args, model, task['test_with_label_data'])
    log.info(']')

    # write prediciton for analysis
    log.info('')
    name = re.findall('.*/(.*).csv', os.path.abspath(args.test_with_label_path))[0]
    eval_out_path = os.path.join(exp_dir, f'output-{name}.csv')
    write_prediction(args, model, task['test_with_label_data'], eval_out_path,
                     mode='analysis', in_path=args.test_with_label_path)
    # debug
    score_task(truth_loc=args.test_with_label_path, prediction_loc=eval_out_path)

''' Prediction '''

if args.do_predict:
    log.info('\n---- Phase: predict -----')
    if not args.do_train:
        model.load_state_dict(torch.load(model_path))
    else:
        model = best_model
    log.info(f'Write prediction for: {args.test_without_label_path}')
    write_prediction(args, model, task['test_without_label_data'], out_path, mode='minimal')
