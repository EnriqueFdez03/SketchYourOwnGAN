#!/bin/bash
python train.py \
--pathNetwork weights/arboles256.pkl \
--batch_size 4 \
--num_epochs 1000000 \
--max_iters 2500 \
--example_each 250 \
--save_network_each 2500 \
--stats_each 250 \
--path_sketch 'bocetos/arboles-simpleDcha' \
--path_images 'imagenes/arboles' \
--example_folder 'experiments' \
--z_vectors 'experiments/z_vectors_arboles.pt' 