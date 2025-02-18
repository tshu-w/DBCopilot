<div align="center">

<h2 id="your-project-name">DBCᴏᴘɪʟᴏᴛ: Natural Language Querying over Massive Database via Schema Routing</h2>

<p>
  <a href="https://edbticdt2025.upc.edu"><img src="http://img.shields.io/badge/EDBT-2025-4b44ce.svg?style=flat-square" alt="Conference" /></a>
  <a href="https://arxiv.org/abs/2312.03463"><img src="http://img.shields.io/badge/arXiv-2312.03463-B31B1B.svg?style=flat-square" alt="Arxiv" /></a>
</p>

</div>

## News

- [2025-02-05] 🎉 Our paper has been accepted at [EDBT 2025](https://edbticdt2025.upc.edu).

## Description

![DBCopilot](https://github.com/tshu-w/DBCopilot/assets/13161779/8212f2ae-f12f-481a-b8ba-b1de9d8bbef9)

The development of natural language interfaces to databases (NLIDB) has been greatly advanced by the advent of large language models (LLMs), which provide an intuitive way to translate natural language (NL) questions into Structured Query Language (SQL) queries. While significant progress has been made in LLM-based NL2SQL, existing approaches face several challenges in real-world scenarios of natural language querying over massive databases. In this paper, we present DBCopilot, a framework that addresses these challenges by employing a compact and flexible copilot model for routing over massive databases. Specifically, DBCopilot decouples schema-agnostic NL2SQL into schema routing and SQL generation. This framework utilizes a lightweight differentiable search index to construct semantic mappings for massive database schemata, and navigates natural language questions to their target databases and tables in a relation-aware joint retrieval manner. The routed schemata and questions are then fed into LLMs for effective SQL generation. Furthermore, DBCopilot introduces a reverse schema-to-question generation paradigm that can automatically learn and adapt the router over massive databases without manual intervention. Experimental results verify that DBCopilot is a scalable and effective solution for schema-agnostic NL2SQL, providing a significant advance in handling natural language querying over massive databases for NLIDB.

## How to run
First, install dependencies
```console
# clone project
git clone --recursive https://github.com/tshu-w/DBCopilot
cd DBCopilot

# [SUGGESTED] use conda environment
conda env create -f environment.yaml
conda activate DBCopilot

# [ALTERNATIVE] install requirements directly
pip install -r requirements.txt
```

Next, download and extract the data from [OneDrive Share Link](https://1drv.ms/u/s!AlCpSo470WIyo-sQPTT1K-mnzpC3fA?e=QISuff).

Finally, run the experiments with the following commands:
```console
# Train the schema questioning models:
./scripts/sweep --config configs/sweep_fit_schema_questioning.yaml

# Training data synthesis
python scripts/synthesize_data.py

# Train the schema routers:
./scripts/sweep --config configs/sweep_fit_schema_routing.yaml

# End-to-end text-to-SQL evaluation:
python scripts/evaluate_text2sql.py
```

You can also train and evaluate a single model with the `run` script.
```console
# fit with the XXX config
./run fit --config configs/XXX.yaml
# or specific command line arguments
./run fit --model Model --data DataModule --data.batch_size 32 --trainer.gpus 0,

# evaluate with the checkpoint
./run test --config configs/XXX.yaml --ckpt_path ckpt_path

# get the script help
./run --help
./run fit --help
```

## Citation
```
@article{wang2023dbcopilot,
  author       = {Tianshu Wang and Hongyu Lin and Xianpei Han and Le Sun and Xiaoyang Chen and Hao Wang and Zhenyu Zeng},
  title        = {DBCopilot: Scaling Natural Language Querying to Massive Databases},
  journal      = {CoRR},
  year         = 2023,
  volume       = {abs/2312.03463},
  doi          = {10.48550/arXiv.2312.03463},
  eprint       = {2312.03463},
  eprinttype   = {arXiv},
  url          = {https://doi.org/10.48550/arXiv.2312.03463},
}
```
