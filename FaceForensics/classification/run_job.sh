#!/bin/bash
#SBATCH -N 1 # node
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=4
#SBATCH -t 23:00:00
#SBATCH --gres=gpu:1

cd /home/hs12/Computer_Vision_algs/FaceForensics/classification

#TRAIN
#python detect_from_video.py -i /tigress/hs12/Computer_Vision/forensics_data/train_vids/FAKE/ -o /tigress/hs12/Computer_Vision/forensics_data/train/FAKE/
#python detect_from_video.py -i /tigress/hs12/Computer_Vision/forensics_data/train_vids/REAL/ -o /tigress/hs12/Computer_Vision/forensics_data/train/REAL/

#VALIDATION
#python detect_from_video.py -i /tigress/hs12/Computer_Vision/forensics_data/val_vids/FAKE/ -o /tigress/hs12/Computer_Vision/forensics_data/validation/FAKE/
#python detect_from_video.py -i /tigress/hs12/Computer_Vision/forensics_data/val_vids/REAL/ -o /tigress/hs12/Computer_Vision/forensics_data/validation/REAL/

#TEST 
#python detect_from_video.py -i /tigress/hs12/Computer_Vision/forensics_data/test_vids/FAKE/ -o /tigress/hs12/Computer_Vision/forensics_data/test/FAKE/
python detect_from_video.py -i /tigress/hs12/Computer_Vision/forensics_data/test_vids/REAL/ -o /tigress/hs12/Computer_Vision/forensics_data/test/REAL/
