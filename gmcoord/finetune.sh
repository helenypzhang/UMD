#!/bin/sh

#SBATCH -o gmcoord.out
#SBATCH -e gmcoord.err
#SBATCH --job-name=gmcoord
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# python prepro/prepro_pretraining_data.py

# bash run_scripts/pretrain_umd.sh


# python prepro/prepro_finetuning_data.py

bash run_scripts/finetune_umd_step2.sh
# bash run_scripts/finetune_umd_roco.sh


# bash run_scripts/test_umd.sh
