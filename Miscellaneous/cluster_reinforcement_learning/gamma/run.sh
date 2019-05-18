#!/bin/bash

for d in ./*/ # first-level nested subdirectories (exp_name)
do
	cd "$d"
	cp ../learn1.py .
	cp ../learn2.py .
	cp ../model.py .
	cp ../q_learning.py .
	cp ../run_learn1.slurm .
	cp ../run_learn2.slurm .
	mkdir -p results # make results dir if not exist
	sbatch run_learn1.slurm # runs program in subdirectory
	sbatch run_learn2.slurm # runs program in subdirectory
	cd ..
done
