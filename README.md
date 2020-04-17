# semeval-humor


[SemEval-2020 Task 7: Assessing Humor in Edited News Headlines](https://competitions.codalab.org/competitions/20970)


## 1 Environment
(only need to configure this for the first time unless otherwise specified)

First, you need to have [Anaconda 3](https://docs.anaconda.com/anaconda/install/) installed. Then create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with required packages specified in `environment.yml`:
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
CACHE=/path/to/cache/directory
bash scripts/path_setup.sh HUGGINGFACE_TRANSFORMERS_CACHE $CACHE
# this will add the following line to ~/.bash_profile (mac) or ~/.bashrc (linux)
# export HUGGINGFACE_TRANSFORMERS_CACHE=/path/to/cache/directory
```

## 2 Data
- see `data` directory
- datasets: [Humicroedit](https://www.aclweb.org/anthology/N19-1012/) (official task data) and [Funlines](https://arxiv.org/pdf/2002.02031.pdf) (additional data)
- you can download the data from the [source website](https://www.cs.rochester.edu/u/nhossain/humicroedit.html), or simply run
  ```bash
  bash scripts/download_data.sh
  ```
  This gives the same data as in `data` directory.


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

To see [tensorboard](https://pytorch.org/docs/stable/tensorboard.html) output:
```bash
open http://localhost:6006
tensorboard --logdir tensorboard_train
# you may need to wait a few seconds and refresh the page

open http://localhost:6006
tensorboard --logdir tensorboard_val
# you may need to wait a few seconds and refresh the page
```

## 4 Scirpts
To get the baseline results, run

```bash
cd scripts
bash baseline.sh >> ../baseline_output/results.log
```
