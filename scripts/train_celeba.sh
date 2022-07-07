#!/bin/bash
python train_repeat.py \
--pathNetwork weights/celeba256.pkl \
--batch_size 4 \
--num_epochs 100000 \
--max_iters 3000 \
--example_each 250 \
--save_network_each 3000 \
--stats_each 250 \
--path_sketch 'bocetos/celeba-2,bocetos/celeba,bocetos/celeba-3,bocetos/celeba-4' \
--path_images imagenes/celeba \
--example_folder "/media/enrique/HDD/TFG/celeba-reg" \
--z_vectors '/media/enrique/HDD/experiments/celeba-2/2022-06-12 21_27_26.128938/z_vectors.pt' \
--repeat 3 \
--enable_aug

