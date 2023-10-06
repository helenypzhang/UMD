num_gpus=1
per_gpu_batchsize=16

# # === VQA === 
# python main.py with data_root=/home/yupei/workspaces/UMD-master/data/finetune_arrows/ \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_finetune_vqa_vqa_rad \
#  per_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=384 \
#  tokenizer=/home/yupei/workspaces/UMD-master/downloaded/roberta-base \
#  load_path=/home/yupei/workspaces/tdcalib_checkpoints/vqa_vqa_rad/epoch=57-step=5567.ckpt


# python main.py with data_root=/home/yupei/workspaces/UMD-master/data/finetune_arrows/ \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_finetune_vqa_slack \
#  per_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=384 \
#  tokenizer=/home/yupei/workspaces/UMD-master/downloaded/roberta-base \
#  load_path=/home/yupei/workspaces/tdcalib_checkpoints/vqa_slack/epoch=23-step=3695.ckpt \
#  clip_resizedcrop

# python main.py with data_root=/home/yupei/workspaces/UMD-master/data/finetune_arrows/ \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_finetune_vqa_medvqa_2019 \
#  per_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=384 \
#  tokenizer=/home/yupei/workspaces/UMD-master/downloaded/roberta-base \
#  load_path=/home/yupei/workspaces/tdcalib_checkpoints/vqa_medvqa_2019/epoch=19-step=5999.ckpt \
#  clip_resizedcrop

# # === CLS ===
python main.py with data_root=/home/yupei/workspaces/UMD-master/data/finetune_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_cls_melinda_p_meth_label \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=/home/yupei/workspaces/UMD-master/downloaded/roberta-base \
 load_path=/home/yupei/workspaces/tdcalib_checkpoints/cls_melinda/epoch=24-step=6799.ckpt \
 clip_resizedcrop
