# Lightning-Template

A clean and scalable template to structure ML paper-code the same so that work can easily be extended and replicated.

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT

---

<div align="center">

<h2 id="your-project-name">Your Project Name</h2>

<p>
<a href="https://arxiv.org/abs/1706.03762"><img src="http://img.shields.io/badge/arxiv-1706.03762-B31B1B.svg" alt="Arxiv" /></a>
<a href="https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"><img src="http://img.shields.io/badge/NeurIPS-2017-4b44ce.svg" alt="Conference" /></a>
</p>

</div>

## Description
What it does

## How to run
First, install dependencies
```console
# clone project
git clone https://github.com/YourGithubName/your-repository-name
cd your-repository-name

# [SUGGESTED] use conda environment
conda env create -f environment.yaml
conda activate lit-template

# [ALTERNATIVE] install requirements directly
pip install -r requirements.txt
```

Next, to obtain the main results of the paper:
```console
# commands to get the main results
```

You can also run experiments with the `run` script.
```console
# fit with the demo config
./run fit --config configs/demo.yaml
# or specific command line arguments
./run fit --model MNISTModel --data MNISTDataModule --data.batch_size 32 --trainer.gpus 0

# evaluate with the checkpoint
./run test --config configs/demo.yaml --ckpt_path ckpt_path

# get the script help
./run --help
./run fit --help
```

## Citation
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
