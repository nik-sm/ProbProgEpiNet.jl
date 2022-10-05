#!/usr/bin/env bash
logdir=test_log
mkdir -p $logdir
export JULIA_PROJECT=. 

noise=5e-3
lr=0.00001
timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S")
git_hash=$(git rev-parse --short HEAD)
if ! git diff-index --quiet HEAD --; then
    git_hash=${git_hash}+
fi

outfile=$logdir/time\=$timestamp,noise\=$noise,lr\=$lr,seed\=$seed,git\=$git_hash

# --track-allocation=user 
time julia scripts/main.jl $@ \
  --tiny \
  --fixed_total_E0=0.005 \
  --obs_noise_std=$noise \
  --lr=$lr \
  --git_hash=$git_hash \
  --county=Test > $outfile 2>&1 

[[ $? -eq 0 ]] && echo 'Test run succeeded!' || { echo 'Test run failed...'; cat $outfile ; }
