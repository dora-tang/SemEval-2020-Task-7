# semeval-humor

## 1 Environment
(only need to configure this for the first time unless otherwise specified)

First, you need to have Anaconda 3 installed. Then create a **conda environment** with required packages by:
```bash
conda env create -f environment.yml
```
activate the environment (do this before running code)
```bash
conda activate humor # or source activate humor
```
deactivate the environment (optionally do this after running code)
```bash
conda deactivate # or source deactivate
```
**spaCy**
```bash
python -m spacy download en
```
**HuggingFace Transformers Cache Directory**
we use the [transformers library by HuggingFace](https://github.com/huggingface/transformers). Save caches so you don't have to download the same model more than once.
```bash

# put cache directory in bash_profile (mac) bashrc (linux)
#CACHE=/share/data/speech/Data/sjin/data/cache/transformers
#CACHE=/Users/jinshuning/Documents/Github/data/cache/transformers
CACHE=[define your cache directory here]
bash scripts/path_setup.sh HUGGINGFACE_TRANSFORMERS_CACHE $CACHE
```

## 2 Data
- see *data* directory
- datasets: **Humicroedit** (official task data) and **Funlines** (additional data)
- you can download the data from the [source website](https://www.cs.rochester.edu/u/nhossain/humicroedit.html), or simply run `bash scripts/download_data.sh`. This is the same data as in data directory.


## 3 Experiment output
```
experiment_directory
├── log.log
├── params.json
# if args.save_model
├── model_state.th
# if args.tensorboard
├── tensorboard_train
├── tensorboard_val
# if args.do_eval
└── output-{eval_data_name}.csv
```

To see tensorboard output:
```bash
open http://localhost:6006
tensorboard --logdir tensorboard_train
# you may need to wait a few seconds and refresh the page

open http://localhost:6006
tensorboard --logdir tensorboard_val
# you may need to wait a few seconds and refresh the page
```
