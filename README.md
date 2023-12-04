<div align="center">

<h2 id="your-project-name">DBCᴏᴘɪʟᴏᴛ: Scaling Natural Language Querying to Massive Databases</h2>
<!--
<p>
<a href="https://arxiv.org/abs/1706.03762"><img src="http://img.shields.io/badge/arxiv-1706.03762-B31B1B.svg" alt="Arxiv" /></a>
<a href="https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"><img src="http://img.shields.io/badge/NeurIPS-2017-4b44ce.svg" alt="Conference" /></a>
</p>
//-->
</div>

## Description

![DBCopilot](https://github.com/tshu-w/DBCopilot/assets/13161779/8212f2ae-f12f-481a-b8ba-b1de9d8bbef9)

Text-to-SQL simplifies database interactions by enabling non-experts to convert their natural language (NL) questions into Structured Query Language (SQL) queries. While recent advances in large language models (LLMs) have improved the zero-shot text-to-SQL paradigm, existing methods face scalability challenges when dealing with massive, dynamically changing databases. This paper introduces DBCopilot, a framework that addresses these challenges by employing a compact and flexible copilot model for routing across massive databases. Specifically, DBCopilot decouples the text-to-SQL process into schema routing and SQL generation, leveraging a lightweight sequence-to-sequence neural network-based router to formulate database connections and navigate natural language questions through databases and tables. The routed schemas and questions are then fed into LLMs for efficient SQL generation. Furthermore, DBCopilot also introduced a reverse schema-to-question generation paradigm, which can learn and adapt the router over massive databases automatically without requiring manual intervention. Experimental results demonstrate that DBCopilot is a scalable and effective solution for real-world text-to-SQL tasks, providing a significant advancement in handling large-scale schemas.

## How to run
First, install dependencies
```console
# clone project
git clone https://github.com/tshu-w/DBCopilot
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
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
