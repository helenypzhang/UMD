#!/bin/sh

#SBATCH -o mrpretrain.out
#SBATCH -e mrpretrain.err
#SBATCH --job-name=mrpretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4

# python prepro/prepro_pretraining_data.py

bash run_scripts/pretrain_umd.sh


# python prepro/prepro_finetuning_data.py

# bash run_scripts/finetune_umd.sh


# bash run_scripts/test_umd.sh
