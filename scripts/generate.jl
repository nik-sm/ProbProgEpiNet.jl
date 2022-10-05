using ProbProgEpiNet

# Begin with model parameters in JSON
args = process_args(parse_commandline())

# Generate CSV files using the fixed parameters
name = get_name(args)
args[:output_dir] = mkpath(joinpath(args[:path_out], name))
open(joinpath(args[:output_dir], "log.txt"), "w") do io
    args[:io] = io
    run_compare(args, false, false)
end

# Condition on the generated data
args[:num_traj] = 100
args[:real_deaths] = joinpath(args[:output_dir], "generated_trajectories_deaths.csv")
args[:real_confirmed] = joinpath(args[:output_dir], "generated_trajectories_confirmed.csv")
println(args[:io], args[:real_deaths])
println(args[:io], args[:real_confirmed])
@assert isfile(args[:real_deaths])
@assert isfile(args[:real_confirmed])

# Store everything in subfolder
# args[:output_dir] = joinpath(args[:output_dir], "validation_results")
# mkpath(args[:output_dir])

open(joinpath(args[:output_dir], "log.txt"), "w") do io
    run_inference(args, io)
    args[:json_results] = joinpath(args[:output_dir], args[:json_results_name])

end

run_compare(args)
