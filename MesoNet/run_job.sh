#!/bin/bash
#SBATCH -N 1 # node
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=4
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:1

cd /home/hs12/Computer_Vision_algs/MesoNet

python example.py
