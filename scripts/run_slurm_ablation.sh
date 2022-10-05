#!/usr/bin/env bash
# Submit a job to slurm
set -euo pipefail

### 

function usage {
    echo "USAGE:"
    echo "bash $0 -c <CLUSTER_NAME> -o <OUTPUT_DIR_SUFFIX> [-t]"
    echo "-t indicates test mode (stop after 1 loop)"
}

test_mode=0
cluster=""
output_dir_suffix=""
while getopts "hc:o:t" opt; do
    case ${opt} in
    h )
        usage
        exit 0
        ;; 
    c )
        cluster=$OPTARG
        ;;
    o )
        output_dir_suffix=$OPTARG
        ;;
    t )
        test_mode=1
        echo TEST MODE
        # Just launches 1 run
        ;;
    * )
        usage
        exit 1
        ;;
    esac
done
shift $((OPTIND - 1))

[[ -z $cluster ]] && { echo Missing required arg: cluster name! >&2; exit 1; }

# project dir must be specified and must exist
if [[ $cluster == "discovery" ]]; then
    project_dir_base=/scratch
elif [[ $cluster == "grid" ]] || [[ $cluster == "supercloud" ]]; then
    project_dir_base=/home/gridsan
else
    echo "Invalid cluster name: $cluster" >&2; exit 1
fi
[[ -d $project_dir_base ]] || { echo no such directory: $project_dir_base >&2; exit 1; }

# output dir defaults to <month>_<day>.
if [[ -z $output_dir_suffix ]]; then 
    output_dir_suffix=$(date +%m_%d)
    echo output dir suffix not given. Defaulting to: $output_dir_suffix 
fi

### 

PROJECT_DIR=${project_dir_base}/$(whoami)/ProbProgEpiNet.jl
LOG_DIR=$PROJECT_DIR/log/${output_dir_suffix}
output_dir_base="output/${output_dir_suffix}"
ENTRYPOINT_SCRIPT=$PROJECT_DIR/scripts/worker_startup.sh

mkdir -p $LOG_DIR
mkdir -p $output_dir_base

git_hash=$(git rev-parse --short HEAD)
if ! git diff-index --quiet HEAD --; then
    git_hash=${git_hash}+
fi

mem=10
seed=1
config=config.json
for county in middlesex-exp losangeles-exp miamidade-exp ; do
for inf_thresh_after_lead in 0.02 0.04 0.08 0.002 ; do
for iters in 20; do # 40; do
for prior_gamma_logit_mean in -1.79; do  # 7 days
for prior_lambda_logit_mean in -2.56; do  # 13.9 days
for lead_in_time in 7; do
for lr in 6e-4; do
for knots in 1; do
for scaling_factor in 5e-5; do
for prior_betaE_logit_mean in -2.64; do
for prior_beta_logit_std_L in -2; do
for noise_fn in 'day_scaling' ; do
for noise in 1e-2 5e-2; do # 5e-3 1e-3 25e-5 5e-4 1e-4 ; do
    fixed_total_E0=$inf_thresh_after_lead

    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S")

    outfile=$LOG_DIR/time\=$timestamp,noise\=$noise,county\=$county,noise_fn\=$noise_fn,E0\=$fixed_total_E0

    cmd_string="/bin/bash $ENTRYPOINT_SCRIPT $PROJECT_DIR $@ \
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
        --duration=163 \
        --experiment_config=$config \
        --output_terms"

    echo Submitting: $cmd_string

    sbatch  \
    --nodes=1 \
    --time=24:00:00 \
    --job-name=param_inference \
    --mem=${mem}Gb \
    --output=$outfile \
    --open-mode=truncate \
    --wrap="$cmd_string"

    [[ $? -eq 0 ]] && echo Submission SUCCESS || echo Submission FAILURE

    sleep 1  # ensures unique name due to timestamp

    if [[ $test_mode -eq 1 ]]; then
        echo TEST MODE DONE
        exit 0
    fi

done
done
done
done
done
done
done
done
done
done
done
done
done
