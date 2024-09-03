DATASET="CIFAR10"
IPC=10
NETWORK="ConvNet"
RUNNAME="DMFYI"
TAGS="CIFAR10_10IPC"
DEVICE='1'

python main_DM.py  --dataset ${DATASET}  --model ${NETWORK}  --ipc ${IPC}  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init real  --lr_img 1  --num_exp 5  --num_eval 5 --run_name ${RUNNAME} --run_tags ${TAGS} --device ${DEVICE} --eval_mode M