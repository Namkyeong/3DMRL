# Molecular Interaction Prediction Tasks


## Dataset Preparation

### Chromophore Property Prediction Task

**Pre-training Datasets**

1. Download Chromophore dataset from https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2
2. Put dataset into `data_pretrain/raw` directory

**Downstream Task Datasets**
1. With the downloaded dataset above, make separate csv file for each dataset **Absorption max (nm)**,  **Emission max (nm)**, and **Lifetime (ns)**. Erase Nan values for each dataset.
2. Put each dataset into the following directory: `data_eval/{dataset_name}/raw`


### Solvation Free Energy Prediction Task

**Pre-training Datasets**

1. Download Solvation Free Energy datasets from https://ars.els-cdn.com/content/image/1-s2.0-S1385894721008925-mmc1.txt
2. Put dataset into `data_pretrain/raw` directory.

**Downstream Task Datasets**
1. Download Solvation Free Energy datasets from https://ars.els-cdn.com/content/image/1-s2.0-S1385894721008925-mmc2.xlsx
2. Create the dataset based on the **Source_all** column in the excel file.
3. Make separate csv file for each data source.
4. Put each dataset into the following directory: `data_eval/{dataset_name}/raw`


When we run the `pretrain.py` or `evaluate.py`, code will automatically create `torch-geometric` data with the prepared datasets.


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