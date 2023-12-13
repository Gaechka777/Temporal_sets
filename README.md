﻿# Identifying the relationship between labels using an attention-based algorithm 


# Requirements
All criteria for libraries are specified in the file "requirements.txt".

# Usage
You can download datasets using the links from the paper.

Install dependencies

```bash
# clone project
git clone https://github.com/Gaechka777/Temporal_sets.git

# install requirements
pip install -r requirements.txt

#Scripts for downloading and preparing diffrent datasets are located in scripts/datasets/. In this code work for Data: demand, gender, liquor, order, sales.

bash scripts/datasets/{name dataset}/prepare_{name dataset}.sh
```
The exception is "order" dataset, for which you need place initial files in corresponding folder mannually.

All models parametrs are specified in configs/base.json and configs/train_params.json.

Before starting the training, change the path to dataset in the configs/base.json.

Run model:
```bash
 bash scripts/run.sh
```
