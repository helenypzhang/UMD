num_gpus=2
per_gpu_batchsize=8

python main.py \
 with data_root=/home/yupei/workspaces/UMD-master/data/pretrain_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_pretrain_umd \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 max_text_len=64 \
 tokenizer=/home/yupei/workspaces/UMD-master/downloaded/roberta-base

# python main.py \
#  with data_root=/home/yupei/workspaces/UMD-master/data/pretrain_arrows/ \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_pretrain_umd \
#  per_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=384 max_text_len=64 \
#  tokenizer=/home/yupei/workspaces/UMD-master/downloaded/roberta-base \
#  load_path=checkpoints/epoch=42-step=72540.ckpt

# python main.py \
#  with data_root=/home/yupei/workspaces/UMD-master/data/pretrain_arrows/ \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_pretrain_umd \
#  per_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=384 max_text_len=64 \
#  tokenizer=/home/yupei/workspaces/UMD-master/downloaded/roberta-base \
#  resume_from=checkpoints/continue/epoch=21-step=37113.ckpt
