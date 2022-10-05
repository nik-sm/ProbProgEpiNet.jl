using ProbProgEpiNet

ENV["GKSwstype"] = "100"

args = process_args(parse_commandline())

if args[:mode] == "infer"
    name = get_name(args)
    println(("name", name))
    args[:output_dir] = mkpath(joinpath(args[:path_out], name))

    open(joinpath(args[:output_dir], "log.txt"), "w") do io
        args[:io] = io
        run_inference(args, io)
    end

    if args[:inf_strat] == "bbvi"
        args[:json_results] = joinpath(args[:output_dir], args[:json_results_name])
        run_compare(deepcopy(args), true)
        run_compare(deepcopy(args), false)
    end

elseif args[:mode] == "compare"
    if args[:inf_strat] != "bbvi"
        error("Not implemented")
    end
    # NOTE - compare puts output in the same dir where the JSON results live
    run_compare(deepcopy(args), true)
    run_compare(deepcopy(args), false)
end
