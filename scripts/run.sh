#!/usr/bin/env bash
set -euo pipefail
export JULIA_PROJECT=.

run_name="time_varying_6_01"

log_dir=log/$run_name
output_dir_base=output/$run_name

mkdir -p $log_dir
mkdir -p $output_dir_base

git_hash=$(git rev-parse --short HEAD)
if ! git diff-index --quiet HEAD --; then
    git_hash=${git_hash}+
fi

seed=1
iters=100
samples_per_iter=120
prior_gamma_logit_mean=-1.79
prior_lambda_logit_mean=-2.56
lead_in_time=7
inf_thresh_after_lead=0.002
lr=6e-4  # We were taking steps according to a sum instead of a mean. The factor of 1 / 120 was absorbed into LR.
scaling_factor=5e-5
prior_betaE_logit_mean=-2.64
noise_fn=day_scaling
county=middlesex-exp
fixed_total_E0=$inf_thresh_after_lead
knots=4
config=config.json

for prior_beta_logit_std_L in -3.0 -2.0; do # narrowed
for noise in 1e-5 5e-5; do

timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S")

outfile=$log_dir/time\=$timestamp,git\=$git_hash

time julia scripts/main.jl \
    --time_varying \
    --obs_noise_std=$noise \
    --iterations=$iters \
    --random_seed=$seed \
    --timestamp=$timestamp \
    --county=$county \
    --fixed_total_E0=$fixed_total_E0 \
    --prior_gamma_logit_mean=$prior_gamma_logit_mean \
    --prior_lambda_logit_mean=$prior_lambda_logit_mean \
    --prior_betaE_logit_mean=$prior_betaE_logit_mean \
    --prior_beta_logit_std_L=$prior_beta_logit_std_L \
    --lead_in_time=$lead_in_time \
    --num_traj=100 \
    --knots=$knots \
    --git_hash=$git_hash \
    --noise_fn=$noise_fn \
    --scaling_factor=$scaling_factor \
    --output_dir_base=$output_dir_base \
    --inf_thresh_after_lead=$inf_thresh_after_lead \
    --final_inf_saturation_frac=0.5 \
    --samples_per_iter=120 \
    --duration=90 \
    --experiment_config=$config \
    --output_terms 2>&1 \
| tee $outfile > /dev/null &

done
done