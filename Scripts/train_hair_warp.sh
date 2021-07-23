# train data dir
train_data_dir='/home/ppxgg/Hair/TrainData'

# test data dir
val_data_dir='/home/ppxgg/Hair/TrainData'

# number of batchSize
batch_size=4

# number of iterations
niter=100000

# of input image channels
input_nc=3

# name of the model
netF="HairWarpNet"

# feature pyramid
pyr_lvls=4

# prediction level

gpu=2
pre_lvls=4

CUDA_VISIBLE_DEVICES=$gpu \
python train_dynamic.py \
--train_data_dir $train_data_dir \
--val_data_dir $val_data_dir \
--batch_size $batch_size \
--niter $niter \
--input_nc $input_nc \
--netF $netF \
--pyr_lvls $pyr_lvls \
--pre_lvls $pre_lvls \
--train_flow

pre_lvls=3

CUDA_VISIBLE_DEVICES=$gpu \
python train_dynamic.py \
--train_data_dir $train_data_dir \
--val_data_dir $val_data_dir \
--batch_size $batch_size \
--niter $niter \
--input_nc $input_nc \
--netF $netF \
--pyr_lvls $pyr_lvls \
--pre_lvls $pre_lvls \
--train_flow

pre_lvls=2

CUDA_VISIBLE_DEVICES=$gpu \
python train_dynamic.py \
--train_data_dir $train_data_dir \
--val_data_dir $val_data_dir \
--batch_size $batch_size \
--niter $niter \
--input_nc $input_nc \
--netF $netF \
--pyr_lvls $pyr_lvls \
--pre_lvls $pre_lvls \
--train_flow

pre_lvls=1

CUDA_VISIBLE_DEVICES=$gpu \
python train_dynamic.py \
--train_data_dir $train_data_dir \
--val_data_dir $val_data_dir \
--batch_size $batch_size \
--niter $niter \
--input_nc $input_nc \
--netF $netF \
--pyr_lvls $pyr_lvls \
--pre_lvls $pre_lvls \
--train_flow