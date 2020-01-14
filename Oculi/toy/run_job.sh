#!/bin/bash                                                                                                                                                                
#SBATCH -N 1 # node                                                                                                                                                        
#SBATCH --mem=128G                                                                                                                                                         
#SBATCH --ntasks-per-node=4                                                                                                                                                
#SBATCH -t 23:00:00                                                                                                                                                         
#SBATCH --gres=gpu:1                                                                                                                                                        

cd /home/hs12/Computer_Vision_algs/oculi/toy

DIR='/tigress/hs12/Computer_Vision/subset_data/training_pipeline/test/REAL'
for f in "$DIR"*
do
    echo f 
    python run_lrcn.py --input_vid_path="$f" >> "${f%}out_real.txt" --out_dir="$f"
done

