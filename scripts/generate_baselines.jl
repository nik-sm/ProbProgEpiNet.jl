using ProbProgEpiNet

# Begin with model parameters in JSON
args = process_args(parse_commandline())

# Generate CSV files using the fixed parameters
run_compare(args, true, false)
