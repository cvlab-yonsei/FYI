DATASET="CIFAR10"
IPC=1
NETWORK="ConvNet"
SYN_STEPS=50
EXPERT_EPOCHS=2
MAX_START_EPOCH=2
LR_IMG=100
LR_LR=1e-7
LR_TEACHER=0.01
BUFFER_PATH="buffer"
DATA_PATH="/dataset"
RUNNAME="MTTFYI"
TAGS="CIFAR10_10IPC"

python distill.py --dataset ${DATASET} --ipc ${IPC} --model ${NETWORK} --syn_steps ${SYN_STEPS} --expert_epochs ${EXPERT_EPOCHS} --max_start_epoch ${MAX_START_EPOCH} --lr_img ${LR_IMG} --lr_lr ${LR_LR} --lr_teacher ${LR_TEACHER} --buffer_path ${BUFFER_PATH} --data_path ${DATA_PATH} --run_name ${RUNNAME} --run_tags ${TAGS} --zca