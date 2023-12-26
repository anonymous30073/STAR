#!/bin/bash

for D in books games tv; do
  for F in output/$D/star_distil; do
      sbatch -p gpu20 --gres gpu:1 --wrap="python main.py --op test --result_folder $F --eval_neg_file f:test_neg_SB_BM25_100";
      sbatch -p gpu20 --gres gpu:1 --wrap="python main.py --op test --result_folder $F --eval_neg_file f:test_neg_standard_100";
      sbatch -p gpu20 --gres gpu:1 --wrap="python main.py --op test --result_folder $F --eval_neg_file f:test_neg_popular_100";
  done;
done