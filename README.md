# scGAT   

## Installation:

Installation Tested on Ubuntu 16.04, CentOS 7, MacOS catalina with Python 3.6.8 on one NVIDIA RTX 2080Ti GPU.

```shell
conda create -n scgat python=3.6.8 pip
conda activate scgat
pip install -r requirements.txt
```

## Quick Start

Take an example of Alzheimer’s disease datasets ([GSE138852](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE138852)) analyzed in the manuscript.

```shell
mkdir GSE138852
wget -P GSE138852/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE138nnn/GSE138852/suppl/GSE138852_counts.csv.gz
```

### 2. Preprocess input files

Go to dataset_stat.ipynb and run preprocessingCSV function. This step generates Use_expression.csv (preprocessed file).  

- **dir** defines file directory(CSV or 10X(default))  
- **datasetName** raw expression matrix data in csv format 
- **csvFilename** output Use_expression.csv file

```
preprocessingCSV("GSE138852/","GSE138852_counts.csv.gz","GSE138852/Use_expression.csv")
```

#### CSV format

Cell/Gene filtering without inferring LTMG:
```shell
python -W ignore PreprocessingscGNN.py --datasetName GSE138852_counts.csv.gz --datasetDir GSE138852/ --LTMGDir GSE138852/ --filetype CSV --geneSelectnum 2000
```

### 3. Run scGNN

We take an example of an analysis in GSE138852. Here we use parameters to demo purposes:

- **batch-size** defines batch-size of the cells for training
- **EM-iteration** defines the number of iteration, default is 10, here we set as 2. 
- **quickmode** for bypassing cluster autoencoder.
- **Regu-epochs** defines epochs in feature autoencoder, default is 500, here we set as 50.
- **EM-epochs** defines epochs in feature autoencoder in the iteration, default is 200, here we set as 20.
- **no-cuda** defines devices in usage. Default is using GPU, add --no-cuda in command line if you only have CPU.
- **regulized-type** (Optional) defines types of regulization, default is noregu for not using LTMG as regulization. User can add --regulized-type LTMG to enable LTMG.

If you want to reproduce results in the manuscript, please use default parameters. 

#### CSV format

For CSV format, we need add **--nonsparseMode**

Without LTMG:
```bash
python -W ignore scGAT.py --datasetName GSE138852 --datasetDir ./  --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --quickmode --nonsparseMode
```

### 4. Check Results

In outputdir now, we have five output files.

- ***_embedding.csv**:    Learned embedding (features) for clustering. Row as cell, col as embeddings. First row as the embedding names (no means). First col as the cell name.

- ***_graph.csv**:        Learned graph edges of the cell graph in tuples: nodeA,nodeB,weights. First row as the name.

- ***_results.txt**:      Identified cell types. First row as the name. 

- ***train_roc_*.csv**:     Dataframe for training ROC output.

- ***train_roc_*.png**:    Plot for training ROC output