DATASET="CIFAR10"
IPC=10
NETWORK="ConvNet"
RUNNAME="DSAFYI"
TAGS="CIFAR10_10IPC"
DEVICE='0'

python main.py  --dataset ${DATASET}  --model ${NETWORK}  --ipc ${IPC}  --init real  --method DSA  --dsa_strategy color_crop_cutout_flip_scale_rotate --run_name ${RUNNAME} --run_tags ${TAGS} --device ${DEVICE} --eval_mode M