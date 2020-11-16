### train
#python main.py --phase train --dataset_dir Drone_dataset --epoch 20 --gpu 0 --n_d 2 --n_scale 2 --checkpoint_dir ./check/drone --sample_dir ./check/drone/sample --continue_train True --L1_lambda 10 --epoch 6 --epoch_step 3 #--load_size 156 --fine_size 128
# Resume at 208k + 5k iterations, one epoch is ~250k  
#python main.py --phase train --dataset_dir Drone_dataset --epoch 20 --gpu 0 --n_d 2 --n_scale 2 --checkpoint_dir ./check/drone --sample_dir ./check/drone/sample --continue_train True --L1_lambda 10 --epoch 3 --epoch_step 1 #--load_size 156 --fine_size 128

#NVIDIA_VISIBLE_DEVICES=0,1
#CUDA_VISIBLE_DEVICES=1
python main.py --phase train --dataset_dir drone --gpu 0 --n_d 2 --n_scale 2 --checkpoint_dir ./check/drone --sample_dir ./check/drone/sample --L1_lambda 10 --use_upsampling 1 --use_demod 1 #--epoch 3 --epoch_step 1 #--load_size 156 --fine_size 128 --continue_train 0 

# Test
#python main.py --phase test --dataset_dir Drone_dataset --gpu 1 --n_d 2 --n_scale 2 --checkpoint_dir ./check/drone --test_dir ./check/drone/testa2b --which_direction AtoB
