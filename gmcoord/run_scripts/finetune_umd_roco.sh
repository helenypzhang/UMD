num_gpus=3
# per_gpu_batchsize=8
per_gpu_batchsize=2

# # === IRTR ===
# python main.py with data_root=/home/yupei/workspaces/UMD-master/data/finetune_arrows/ \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_finetune_irtr_roco get_recall_metric=True \
#  per_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=384 \
#  tokenizer=/home/yupei/workspaces/UMD-master/downloaded/roberta-base \
#  test_only=True \
#  load_path=/home/yupei/workspaces/tdcalib_checkpoints/irtr_roco/epoch=15-step=4319.ckpt

python main.py with data_root=/home/yupei/workspaces/UMD-master/data/finetune_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_irtr_roco get_recall_metric=False \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=/home/yupei/workspaces/UMD-master/downloaded/roberta-base \
 load_path=/home/yupei/workspaces/tdcalib_checkpoints/irtr_roco/epoch=17-step=4859.ckpt \
 clip_resizedcrop