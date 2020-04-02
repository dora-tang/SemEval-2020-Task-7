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
#HUGGINGFACE_TRANSFORMERS_CACHE=/share/data/speech/Data/sjin/data/cache/transformers
#HUGGINGFACE_TRANSFORMERS_CACHE=/Users/jinshuning/Documents/Github/data/cache/transformers
HUGGINGFACE_TRANSFORMERS_CACHE=[define your cache directory here]
case "$(uname -s)" in
  Darwin)
    echo export HUGGINGFACE_TRANSFORMERS_CACHE="$HUGGINGFACE_TRANSFORMERS_CACHE" >> ~/.bash_profile
    source ~/.bash_profile
  ;;
  Linux)
    echo export HUGGINGFACE_TRANSFORMERS_CACHE="$HUGGINGFACE_TRANSFORMERS_CACHE" >> ~/.bashrc
    source ~/.bashrc
  ;;
  *)
    echo 'Automatic path setup is only configured for MacOS and Linux.'
  ;;
esac
```

## 2 Data
- see *data* directory
- datasets: **Humicroedit** (official task data) and **Funlines** (additional data)
- source: https://www.cs.rochester.edu/u/nhossain/humicroedit.html
