# train data dir
train_data_dir='../Train'

# test data dir
val_data_dir='../Val'

# number of batchSize
batch_size=4

# number of iterations
niter=150000

# of input image channels
input_nc=3

# name of the model
netF="HairTempNet"

# input window size
flow_in_wsz=2

gpu=1

CUDA_VISIBLE_DEVICES=$gpu \
python train_dynamic.py \
--train_data_dir $train_data_dir \
--val_data_dir $val_data_dir \
--batch_size $batch_size \
--niter $niter \
--input_nc $input_nc \
--netF $netF \
--flow_in_wsz $flow_in_wsz \
--train_flow

#--continue_train
