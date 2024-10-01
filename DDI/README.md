# Drug-Drug Interaction Interaction Prediction Tasks


## Dataset Preparation

For drug-drug interaction tasks, we provide the dataset used during pre-training and downstream tasks in `data_pretrain/DDI/raw` and `data_eval/{dataset name}/raw`, respectively.
All these datasets are obtained from https://github.com/isjakewong/MIRACLE/tree/main/MIRACLE/datachem.

## Run Pre-training

**[Option 1]** Train model with shell script
```
sh ./scripts/pretrain.sh
```

**[Option 2]** Train model without shell script
```
python pretrain.py
```

Following hyperparameters can be passed into `pretrain.py`:

``dataset``: choose the dataset for model pre-training.

``use_subset``: use the subset of molecule pairs.

``ratio``: choose how much data to use in \%.

``alpha``: hyperparameter that controls the weight of force prediction loss ($\alpha$ in Eq. (7))

After pre-training the model, model check points will be saved in `pretrained_weights/`.


## Downstream tasks

**[Option 1]** Evaluate model with shell script
```
sh ./scripts/downstream.sh
```

**[Option 2]** Train model without shell script
```
python evaluate.py
```