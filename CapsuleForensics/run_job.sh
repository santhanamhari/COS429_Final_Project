#!/bin/bash
#SBATCH -N 1 # node
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=4
#SBATCH -t 23:00:00
#SBATCH --gres=gpu:1

cd /home/hs12/Computer_Vision_algs/Capsule_Forensics_StandAlone
#train = '/home/hs12/Computer_Vision_algs/Capsule_Forensics_StandAlone/dataset/train'
#validation = '/home/hs12/Computer_Vision_algs/Capsule_Forensics_StandAlone/dataset/validation'
#test = /home/hs12/Computer_Vision_algs/Capsule_Forensics_StandAlone/dataset/test

python train.py --dataset dataset --train_set train --val_set validation --outf checkpoints --batchSize 100 --niter 100

