#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 7-00:00:00

python train.py | tee logs_Rain200H.txt