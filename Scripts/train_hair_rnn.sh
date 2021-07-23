# train data dir
train_data_dir='../Train'

# test data dir
val_data_dir='../Val'

# number of batchSize
batch_size=4

# number of iterations
niter=200000

# of input image channels
input_nc=3

# of input size
input_size=2

# name of the model
netG="HairRNNNet"

gpu=3
CUDA_VISIBLE_DEVICES=$gpu \
python train_dynamic.py \
--train_data_dir $train_data_dir \
--val_data_dir $val_data_dir \
--batch_size $batch_size \
--niter $niter \
--input_nc $input_nc \
--input_size $input_size \
--netG $netG

#--continue_train
