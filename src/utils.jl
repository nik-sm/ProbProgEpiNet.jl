function reset_seed(s)
    Random.seed!(s)
    return MersenneTwister(s)
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "mode"
        default = "infer" # infer or compare
        required = false
        "--tiny"
        action = :store_true
        "--multithreaded"
        action = :store_true
        "--iterations"
        arg_type = Int
        default = 30
        "--samples_per_iter"
        arg_type = Int
        default = 120
        "--lr"
        arg_type = Float64
        default = 5e-6
        "--decay"
        arg_type = Int
        default = 10
        "--obs_noise_std"
        arg_type = Float64
        default = 1e-3
        "--random_seed"
        arg_type = Int
        default = 2020
        "--timestamp"
        arg_type = String
        default = string(now())
        "--json_results"
        arg_type = String
        "--experiment_config"
        arg_type = String
        default = "./config.json"
        "--county"
        arg_type = String
        default = "Test"
        "--num_traj"
        arg_type = Int
        default = 100
        "--git_hash"
        arg_type = String
        "--prior_E0_logit_mean"  # This is the mean of the logit of E0
        arg_type = Float64
        default = -7.0
        "--prior_E0_logit_std_L"
        arg_type = Float64
        default = 0.0
        "--fixed_total_E0" # If used, we sample each community's E0, then rescale so the weighted sum equals this value
        arg_type = Float64
        default = -1.0 # sentinal value, replaced by nothing
        "--prior_betaE_logit_mean"
        arg_type = Float64
        default = -1.39
        "--prior_beta_logit_std_L"
        arg_type = Float64
        default = 0.0
        "--prior_gamma_logit_mean"
        arg_type = Float64
        default = logit(6.7e-2) # numbers taken from Lucas
        "--prior_lambda_logit_mean"
        arg_type = Float64
        default = logit(1.1e-1)
        "--prior_gamma_lambda_logit_std_L"
        arg_type = Float64
        default = log(0.1) #  +/- 1 stdev corresponds to incubation times of (13.6, 16.4) days
        "--lead_in_time"
        arg_type = Int
        default = 7 # previously we used 32
        "--inf_thresh_after_lead"
        arg_type = Float64
        default = 0.0025 # previous logic would correspond to ~0.0025
        "--duration"
        arg_type = Int
        default = 163
        # "--start_day"
        #     arg_type = Int
        #     default = 38
        # "--end_day"
        #     arg_type = Int
        #     default = 200
        "--final_inf_saturation_frac"
        arg_type = Float64  # between 0 and 1
        default = 0.5 # by default, final day of data is adjusted to 50% of population size
        "--knots"
        arg_type = Int
        default = 6 # slightly more than 1 per 30 days (using [38:200])
        "--validation_type"
        arg_type = String
        default = nothing
        "--output_dir_base"
        arg_type = String
        default = "output"
        "--noise_fn"
        arg_type = String
        default = "day_scaling"
        "--cond_on_deaths"
        action = :store_true
        "--scaling_factor"
        arg_type = Float64
        default = 1e-5
        "--output_terms"
        action = :store_true
        "--pct_downsample_graph"
        default = -1.0  # -1 will cause no downsampling
        arg_type = Float64
        "--time_varying"
        action = :store_true
        "--time_varying_edges_path"
        default = "time_varying/middlesex"
        "--inf_strat"
        arg_type = String
        default = "bbvi"  # options: bbvi, is, mh
        "--num_chains"
        arg_type = Int
        default = 10
    end

    return parse_args(s, as_symbols=true)
end

function process_args(args)
    delete!(args, :prior_gamma_lambda_logit_std_L)
    # NOTE - JSON.parsefile returns a dictionary with string keys, not symbol
    experiment_conf = JSON.parsefile(args[:experiment_config])
    county_conf = experiment_conf[args[:county]]

    ##Check to see if "death_undercount_factor" is present
    if "death_undercount_factor" ∉ keys(county_conf)
        county_conf["death_undercount_factor"] = 1.0
    end

    if args[:time_varying]
        dates = ["03-16", "03-23", "03-30", "04-06", "04-13", "04-20", "04-27", "05-04", "05-11", "05-18", "05-25", "06-01", "06-08"]
        # Store the first item. We learn graph size by loading this one.
        args[:path_node_attributes] = joinpath(@__DIR__, "..", args[:time_varying_edges_path], "node_attributes.json")
        args[:path_edge_attributes] = joinpath(@__DIR__, "..", args[:time_varying_edges_path], dates[1], "edge_attributes.json")

        # Store other edge file paths
        args[:time_varying_edge_files] = [joinpath(@__DIR__, "..", args[:time_varying_edges_path], d, "edge_attributes.json") for d in dates]
    else
        args[:path_node_attributes] = joinpath(@__DIR__, "..", county_conf["node_attributes"])
        args[:path_edge_attributes] = joinpath(@__DIR__, "..", county_conf["edge_attributes"])
    end

    args[:county_pop] = county_conf["county_pop"]
    args[:jhu_county_name] = county_conf["jhu_county_name"]
    args[:jhu_state_name] = county_conf["jhu_state_name"]

    args[:death_undercount_factor] = county_conf["death_undercount_factor"]
    args[:county_mortality_rate] = county_conf["mortality_rate"]

    args[:path_out] = joinpath(@__DIR__, "..", args[:output_dir_base])
    args[:path_data] = joinpath(@__DIR__, "..", "COVID-19/csse_covid_19_data/csse_covid_19_time_series")
    args[:real_confirmed] = joinpath(args[:path_data], "time_series_covid19_confirmed_US.csv")
    args[:real_deaths] = joinpath(args[:path_data], "time_series_covid19_deaths_US.csv")

    isfile(args[:path_node_attributes]) || error("bad path: $(args[:path_node_attributes])")
    isfile(args[:path_edge_attributes]) || error("bad path: $(args[:path_edge_attributes])")

    if args[:fixed_total_E0] < 0
        args[:fixed_total_E0] = nothing
    end

    if args[:pct_downsample_graph] < 0
        args[:pct_downsample_graph] = nothing
    end

    args[:json_results_name] = "posterior_params.json"
    args[:json_results_checkpoint_name] = "checkpoint_posterior_params.json"

    return args
end

function get_name(args)
    name = "time=$(args[:timestamp]),"
    name *= "git_hash=$(args[:git_hash]),"
    name *= "noise=$(round(args[:obs_noise_std], digits=5)),"
    name *= "lr=$(args[:lr]),"
    name *= "lead_in=$(args[:lead_in_time]),"
    name *= "inf_thresh=$(args[:inf_thresh_after_lead]),"
    name *= "duration=$(args[:duration]),"
    name *= "inf_sat=$(args[:final_inf_saturation_frac]),"
    name *= "iters=$(args[:iterations]),"
    if args[:fixed_total_E0] !== nothing
        name *= "fixed_E0=$(args[:fixed_total_E0]),"
    end
    name *= "seed=$(args[:random_seed]),"
    name *= "county=$(args[:county]),"
    if args[:cond_on_deaths]
        name *= "CondOnD,"
    end
    name *= "noise_fn=$(args[:noise_fn]),"
    name *= "sc=$(round(args[:scaling_factor], digits=7)),"
    return name
end
