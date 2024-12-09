#!/bin/bash

#SBATCH --job-name=GNN_VAEGATVAE
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-user=zetong.zhao@yale.edu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
#SBATCH --partition=day
#SBATCH --time=6:00:00

# module load CUDA/11.8.0
# module load cuDNN/8.7.0.84-CUDA-11.8.0

module load miniconda

conda activate scgnnEnv

cd /gpfs/gibbs/project/gerstein/zz465/gnn

python -W ignore scGNN.py --datasetName GSE138852 --datasetDir ./  --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --quickmode --nonsparseMode
