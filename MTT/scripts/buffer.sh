DATASET="CIFAR10"
NETWORK="ConvNet"
TRAIN_EPOCHS=50
NUM_EXPERTS=100
BUFFER_PATH="buffer"
DATA_PATH="/dataset"

python buffer.py --dataset ${DATASET} --model ${NETWORK} --train_epochs ${TRAIN_EPOCHS} --num_experts ${NUM_EXPERTS} --buffer_path ${BUFFER_PATH} --data_path ${DATA_PATH} --zca