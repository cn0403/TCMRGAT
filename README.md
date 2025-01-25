# TCMRGAT
Python 3.8.17 

# Installation
```
conda create -n TCMRGAT python=3.8.17
conda activate TCMRGAT
pip install -r requirements.txt
```


# Dataset
The data used in this paper are all in the data folder, which can be decompressed by the following command.

```
unzip raw_data/raw_data.zip
```

`target_info.csv`
It is the target data we collected.
`ingredient-target_relation.csv`
It's the data set of ingredient target relationship data that we collected.
`ingredient_info.csv`
It is the ingredient data we collected.
`herb-ingredient_relation.csv`
It's the data set of herb ingredient relationship data that we collected.
`herb_feature.csv`
It's the data set of herb feature that we collected.
`formula_herb_relation.csv`
It's the data set of formula herb relationship data that we collected.

# Data Enhancement

For data enhancement, please input following in terminal.

```
python generate_data.py
```

There are some optional parameters.

`--formula_efficacy_name`
The efficacy name of the prescription for data enhancement.
`--generate_num`
Number of generated samples.
`--edge_thr`
The adjacency matrix is used as an edge threshold.

For example

```
python generate_data.py --formula_efficacy_name 清热 --generate_num 2000 --edge_thr 0.8
```

# Training
To train TCMRGAT, please input following in terminal.

```
python main.py
```

There are some optional parameters.

`--formula_efficacy_name`
The name of the predicted formula.
`--is_over_sample`
Whether the prediction is the data after the enhancement.
`--in_feats`
The length of the input feature.
`--hidden_feats`
The length of the hidden layer feature.
`--max_epochs`
Epochs of training.
`--split`
Split of training data
`--lr`
Learning rate in training process

For example, to train TCMRGAT

```
python main.py --formula_efficacy_name 清热 --is_over_sample False --in_feats 1024 --hidden_feats 512 --max_epochs 200 --split 10 --lr 1e-3
```