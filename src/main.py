from base import *
#from models import *
from code.score_task_1 import score_task_1


def parse_args(cl_args):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-bsz', type=int, default=32, help='param: batch size')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-epochs', type=int, default=10, help='max epochs')
    # prevent gradient norm from being too large
    parser.add_argument('-grad_norm', type=float, default=5, help='gradient norm clipping')
    parser.add_argument('-model', type=str, default='cbow',
                        choices=['cbow', 'transformer', ], help='')
    # chocies: 'bert-base-uncased', 'bert-large-uncased','roberta-base', 'roberta-large', 'roberta-large-mnli'
    parser.add_argument('-transformer', type=str, default='bert-base-uncased', help='')
    # parser.add_argument('-finetune', action='store_true', help='')
    parser.add_argument('-finetune', type=int, default=0, choices=[0, 1],help='')
    parser.add_argument('-train_extra', type=int, default=0, choices=[0, 1],
                        help='use funlines dataset as additional dataset if 1')
    #parser.add_argument('-train_extra', action='store_true', help='')
    parser.add_argument('-feature',  type=str, default='edit_context',
                        choices=['edit-context', 'edit-original', ], help='')
    parser.add_argument('-exp',  type=str, default='exp', help='experiment directory')
    # choose pipeline
    parser.add_argument('-do_train', action='store_true', help='')
    parser.add_argument('-do_eval', action='store_true', help='')
    parser.add_argument('-do_predict', action='store_true', help='')
    # additional
    parser.add_argument('-track', action='store_true', help='track training time')
    parser.add_argument('-save_model', action='store_true', help='save model state')
    # paths
    parser.add_argument('-train_path', type=str, default="../data/task-1/train.csv", help='')
    parser.add_argument('-train_extra_path', type=str,
                        default="../data/task-1/train_funlines.csv", help='')
    parser.add_argument('-val_path', type=str, default="../data/task-1/dev.csv", help='')
    parser.add_argument('-test_without_label_path', type=str,
                        default="../data/task-1/test_without_label.csv", help='')
    parser.add_argument('-test_with_label_path', type=str,
                        default="../data/task-1/test_with_label.csv", help='')
    # parser.add_argument('-cache', type=str, default='cache/transformers',
    #                     help='cache directory for pretrained transformers')
    # parser.add_argument('-model_path', action='store_true', help='')
    args = parser.parse_args(cl_args)
    return args


args = parse_args(sys.argv[1:])
seed_all(args.seed)
#cache_dir = args.cache
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dir = os.path.join('../exp1', args.exp)
config_path = os.path.join(dir, 'params.conf')
#model_path = os.path.join(dir, 'task-1-model.pth')
model_path = os.path.join(dir, 'model_state.th')
out_path = os.path.join(dir, 'task-1-output.csv')
log_path = os.path.join(dir, 'log.log')

init_log()
# if args.phase == 'train':
if args.do_train:
    # clear experiment directory
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)

    log.getLogger().addHandler(log.FileHandler(log_path, mode='w'))
    log.info('---- Phase: train -----')
    with open(config_path, 'w') as conf:
        args_dict = copy.deepcopy(vars(args))
        args_exclude = ['exp', 'track', 'save_model', 'do_eval', 'do_train', 'do_predict',
                        'train_path', 'train_extra_path', 'val_path', 'test_with_label_path', 'test_without_label_path']
        for k in args_exclude:
            del args_dict[k]

        json.dump(args_dict, conf, indent=2)
else:
    # assert experiment directory exists
    assert os.path.isdir(dir)
    assert os.path.exists(config_path)
    assert os.path.exists(model_path)

    log.getLogger().addHandler(log.FileHandler(log_path, mode='a'))
    # log.info(f'\n---- Phase: {args.phase} -----')
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

# log.info(' '.join(sys.argv))
log.info(f'Arguments: {args_json}')
log.info(f'Device: {device}')


''' Preprocessing (Model Specific) '''

if args.model == 'cbow':
    TEXT = data.Field(tokenize='spacy', sequential=True, batch_first=True, use_vocab=True)
    preprocess = 'cbow'
    mask_token = None

elif args.model == 'transformer':
    model_name = args.transformer
    MODELS_DICT = init_transformers()
    model_class, tokenizer_class = MODELS_DICT[model_name]
    try:
        cache_dir = os.environ['HUGGINGFACE_TRANSFORMERS_CACHE']
    except KeyError as e:
        log.info('ERROR: environment variable HUGGINGFACE_TRANSFORMERS_CACHE not found'
                 + '\n\tmust define cache directory for huggingface transformers'
                 + '\n\tRUN THIS: export HUGGINGFACE_TRANSFORMERS_CACHE=/your/path'
                 + '\n\tyou should also append the line to ~/.bashrc (linux) or ~/.bash_profile (mac)'
                 + '\n')
        raise
    tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=cache_dir)
    TEXT = data.Field(tokenize=lambda x: tokenize_and_cut(x, tokenizer=tokenizer),
                      batch_first=True,
                      use_vocab=False,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      # init_token = init_token_idx,
                      # eos_token = eos_token_idx,
                      pad_token=tokenizer.pad_token_id,
                      unk_token=tokenizer.unk_token_id,)
    preprocess = 'transformer'
    mask_token = tokenizer.mask_token


""" Load Data """


def read_task(train_path=None,
              val_path=None,
              test_with_label_path=None,
              test_without_label_path=None):
    task = {}
    log.info('')

    for split_name, path in[('train_data', 'train_path'), ('val_data', 'val_path'),
                            ('test_with_label_data', 'test_with_label_path'),
                            ('test_without_label_data', 'test_without_label_path')]:
        path = eval(path)
        if path is not None:
            log.info(f'read {split_name} from {path}')
            if split_name == 'train_data':
                split = pd.concat([pd.read_csv(i) for i in path])
            else:
                split = pd.read_csv(path)

            task[f'{split_name}'] = get_dataset(
                csv_data=split, text_field=TEXT, preprocess=preprocess, mask_token=mask_token)

    for split_name in task.keys():
        length = len(task[f'{split_name}'])
        log.info(f'Number of examples in {split_name}: {length}')

    return task


if args.train_extra:
    train_path_list = [args.train_path, args.train_extra_path]
else:
    train_path_list = [args.train_path]
val_path = test_with_label_path = test_without_label_path = None
if args.do_train:
    val_path = args.val_path
if args.do_eval:
    test_with_label_path = args.test_with_label_path
if args.do_predict:
    test_without_label_path = args.test_without_label_path
task = read_task(train_path=train_path_list, val_path=val_path,
                 test_without_label_path=test_without_label_path,
                 test_with_label_path=test_with_label_path)


''' Define Model '''

if args.model == 'cbow':
    TEXT, embedding = load_word_embedding(TEXT, task['train_data'])

    embedding.weight.requires_grad = bool(args.finetune)

    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model = CBOW(embedding, PAD_IDX, feature=args.feature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

elif args.model == 'transformer':
    transformer = model_class.from_pretrained(
        model_name, cache_dir=cache_dir, output_hidden_states=True)
    model = PretrainedTransformer(transformer,
                                  finetune=bool(args.finetune),
                                  feature=args.feature,
                                  pad_token_id=tokenizer.pad_token_id,
                                  sep_token_id=tokenizer.sep_token_id,)
    optimizer = AdamW(model.parameters())


''' Training '''
if args.do_train:
    log.info(f'The model has {count_parameters(model)} trainable parameters')
    # log.info(model)
    best_model = train_loop(args, model, optimizer,
                            train_data=task['train_data'], val_data=task['val_data'])

    #log.info('\nTest RMSE of best epoch on validation data')
    log.info('\n[validation result')
    calc_rmse(args, best_model, task['val_data'])
    log.info(']')

    # save model
    if args.save_model:
        torch.save(best_model.state_dict(), model_path)
        log.info(f"\nSave best model state to {model_path}")
    else:
        log.info('\nNOT save model state.')


if args.do_eval:
    log.info('\n---- Phase: eval -----')
    # log.info('START EVAL')
    if not args.do_train:
        model.load_state_dict(torch.load(model_path))
    else:
        model = best_model
    log.info(f'Evaluate: {args.test_with_label_path}')

    # eval
    #log.info('\nevaluate test-with-label data')
    log.info('\n[test result')
    calc_rmse(args, model, task['test_with_label_data'])
    log.info(']')

    # write prediciton for analysis
    log.info('')
    name = re.findall('.*/(.*).csv', os.path.abspath(test_with_label_path))[0]
    eval_out_path = os.path.join(dir, f'output-{name}.csv')
    write_prediction(args, model, task['test_with_label_data'], eval_out_path,
                     mode='analysis', in_path=test_with_label_path)
    # debug
    score_task_1(truth_loc=args.test_with_label_path, prediction_loc=eval_out_path)
    # log.info('END EVAL')

if args.do_predict:
    log.info('\n---- Phase: predict -----')
    if not args.do_train:
        model.load_state_dict(torch.load(model_path))
    else:
        model = best_model
    log.info(f'Write prediction for: {args.test_without_label_path}')
    write_prediction(args, model, task['test_without_label_data'], out_path, mode='minimal')
