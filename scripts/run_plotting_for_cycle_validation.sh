#!/usr/bin/env bash
# This script just deals with the nested output folder structure for the cycle_validation experiment.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source ${SCRIPT_DIR}/../venv/bin/activate

data_dir=/data/shared/coanet_results/cycle_validation_6_07

for county in losangeles-exp miamidade-exp middlesex-exp; do
    for scenario in lo med hi early-peak mid-peak late-peak; do
        echo $county $scenario
        python ${SCRIPT_DIR}/generate_baselines_plots.py --data-path ${data_dir}/${county}/${scenario}
        echo
    done
done

echo OUTPUTS:
ls ${data_dir}/*/*/results/results.csv