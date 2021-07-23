# test data dir
test_data_dir='/home/ubuntu/LC/data'

# test video dir
test_video='video0001'

# number of batchSize
batch_size=1

# of input image channels
input_nc=3

# name of the model
netG="HairSpatNet"

# name of the model
netF="HairWarpNet"

# feature pyramid
pyr_lvls=3

# prediction level
pre_lvls=1

# build temporal coherence window size
build_tc_wsz=5

# start frame
test_start_frame=0

# test frames
test_frames=100

python test_dynamic.py \
--train_data_dir $train_data_dir \
--val_data_dir $val_data_dir \
--batch_size $batch_size \
--input_nc $input_nc \
--netG $netG \
--netF $netF \
--pyr_lvls $pyr_lvls \
--pre_lvls $pre_lvls \
--build_tc_wsz $build_tc_wsz \
--test_start_frame $test_start_frame \
--test_frames $test_frames