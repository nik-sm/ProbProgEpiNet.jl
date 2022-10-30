#!/bin/bash
set -euxo pipefail


# Check for python environment with pandas
rc=0
python3 -c "import pandas" > /dev/null 2>&1 || rc=$?
if [[ $rc -ne 0 ]]; then
  read -p "Pandas required. Install using $(which pip3)? (y/n) " -n 1 -r reply
  echo
  if [[ $reply =~ ^[Yy]$ ]]; then
    pip3 install pandas
  else
    echo "Ok, exiting. Install pandas and retry" >&2
    exit 1
  fi
fi

# Fetch JHU CSSE COVID-19 Data files from https://github.com/CSSEGISandData/COVID-19
# To force fetching from scratch, remove the COVID-19 folder
data_commit=51703402ef39055cc4ce264527b8cfa280fd7682
dir_prefix=csse_covid_19_data/csse_covid_19_time_series
if [[ ! -f COVID-19/$dir_prefix/time_series_covid19_confirmed_US.csv ]]; then
  wget https://github.com/CSSEGISandData/COVID-19/raw/$data_commit/$dir_prefix/time_series_covid19_confirmed_US.csv --directory-prefix COVID-19/$dir_prefix
fi
if [[ ! -f COVID-19/$dir_prefix/time_series_covid19_deaths_US.csv ]]; then
  wget https://github.com/CSSEGISandData/COVID-19/raw/$data_commit/$dir_prefix/time_series_covid19_deaths_US.csv --directory-prefix COVID-19/$dir_prefix
fi

# Unpack network topology data
# To force unpacking, remove ExperimentData, size_varying, and/or time_varying folders
if [[ ! -d ExperimentData ]] || [[ ! -d size_varying ]] || [[ ! -d time_varying ]]; then
  tar xvf data.tgz
fi

# Setup julia environment
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.build(); using Pandas'
