function plot_SEIR(daily_counts)
    fig = Plots.plot()
    first_flag = true
    for daily_counts_iter in daily_counts
        label = ["" "" "" ""]
        if first_flag
            label = ["S" "E" "I" "R"]
            first_flag = false
        end
        data = hcat([[x.S, x.E, x.I, x.R] for x in daily_counts_iter[2:end]]...)
        fig = Plots.plot!(data', label=label, alpha=0.3, linecolor=["blue" "yellow" "red" "green"],
                          title="SEIR Curve", left_margin=50Plots.px)
    end
    return fig
end

function plot_inf(prior_daily_counts, post_daily_counts, obs_counts)
    fig = Plots.plot()
    first_flag = true
    for (prior_daily_counts_iter, post_daily_counts_iter) in zip(prior_daily_counts, post_daily_counts)
        labels = ["" "" ""]
        if first_flag
            labels = ["prior" "posterior" "observation"]
            first_flag = false
        end
        Plots.plot!(cumsum([x.EI for x in prior_daily_counts_iter[2:end]]),
            label=labels[1], linecolor="blue", title="Cumulative Inf", alpha=0.5, left_margin=50Plots.px,
            legend=:topleft)
        Plots.plot!(cumsum([x.EI for x in post_daily_counts_iter[2:end]]),
            label=labels[2], linecolor="red", title="Cumulative Inf", alpha=0.5, left_margin=50Plots.px,
            legend=:topleft)
        fig = Plots.plot!(obs_counts, label=labels[3], alpha=0.5, linecolor="black", left_margin=50Plots.px,
            legend=:topleft)
    end
    return fig
end

function summarize_betas(args, io, name, post_βE_logit_means, post_βE_logit_std_Ls, days)
    println(io, name * " summary")
    knot_days = push!(map(x -> convert(Int, floor(x)), collect(1:days / (args[:knots] - 1):days)), days)

    β = []
    for knot in 1:args[:knots]
        μ = post_βE_logit_means[knot]
        σ = exp(post_βE_logit_std_Ls[knot])
        push!(β, mean(map(x -> sigmoid(x), rand(Normal(μ, σ), 100))))
    end

    spline_β = Spline1D(knot_days, β, k=1)

    daily_β = [spline_β(day) for day in 1:days]

    println(io, "Knot means: $(β)")
    println(io, "Average: $(mean(daily_β))")
    println(io, "Average first 32 days: $(mean(daily_β[1:32]))")
    println(io, "Highest 32 day average: $(maximum(rollmean(daily_β, 32)))")
end

function output_guide_params(args, io, community_sizes, days, json_name="", addl_text="")
    if json_name == ""
        json_name = args[:json_results_name]
    end
    # Fetch learned params from guide
    post_E0_logit_means = get_param(guide, :E0_logit_means)
    post_E0_logit_std_L = get_param(guide, :E0_logit_std_L)

    post_βE_logit_means = get_param(guide, :βE_logit_means)
    post_βE_logit_std_Ls = get_param(guide, :βE_logit_std_Ls)

    post_βI_logit_means = get_param(guide, :βI_logit_means)
    post_βI_logit_std_Ls = get_param(guide, :βI_logit_std_Ls)

    # post_γ_logit_mean = get_param(guide, :γ_logit_mean)
    # post_γ_logit_std_L = get_param(guide, :γ_logit_std_L)

    # post_λ_logit_mean = get_param(guide, :λ_logit_mean)
    # post_λ_logit_std_L = get_param(guide, :λ_logit_std_L)

    # Prior on all other params:
    println(io, addl_text)


    if args[:fixed_total_E0] != nothing
        println(io, ("Fixed total E0", args[:fixed_total_E0]))
        print_normalized_E0_summary(args, io, "Normalized_E0",
            args[:prior_E0_logit_mean], args[:prior_E0_logit_std_L],
            post_E0_logit_means, post_E0_logit_std_L * ones(length(post_E0_logit_means)), community_sizes)
    else
        print_param_summary(io, "E0",
        args[:prior_E0_logit_mean], args[:prior_E0_logit_std_L],
        post_E0_logit_means, post_E0_logit_std_L * ones(length(post_E0_logit_means)))
        fig = Plots.bar(map(sigmoid, post_E0_logit_means))
        xticks!(1:length(community_sizes))
        savefig(fig, joinpath(args[:output_dir], "E0_medians.png"))
    end

    print_param_summary(io, "βE",
      args[:prior_betaE_logit_mean], args[:prior_beta_logit_std_L],
      post_βE_logit_means, post_βE_logit_std_Ls)


    summarize_betas(args, io, "βE", post_βE_logit_means, post_βE_logit_std_Ls, days)

    print_param_summary(io, "βI",
      prior_βI_logit_mean, args[:prior_beta_logit_std_L],
      post_βI_logit_means, post_βI_logit_std_Ls)

    summarize_betas(args, io, "βI", post_βI_logit_means, post_βI_logit_std_Ls, days)

    # print_param_summary(io, "γ",
    #  args[:prior_gamma_logit_mean], args[:prior_gamma_lambda_logit_std_L],
    #  [post_γ_logit_mean], [post_γ_logit_std_L])

    # print_param_summary(io, "λ",
    #  args[:prior_lambda_logit_mean], args[:prior_gamma_lambda_logit_std_L],
    #  [post_λ_logit_mean], [post_λ_logit_std_L])

    # Save posterior params to JSON file
    posterior_params = Dict(
      "post_E0_logit_means" => post_E0_logit_means,
      "post_E0_logit_std_L" => post_E0_logit_std_L,

      "fixed_total_E0" => args[:fixed_total_E0],

      "post_βE_logit_means" => post_βE_logit_means,
      "post_βE_logit_std_Ls" => post_βE_logit_std_Ls,

      "post_βI_logit_means" => post_βI_logit_means,
      "post_βI_logit_std_Ls" => post_βI_logit_std_Ls,

      # "post_γ_logit_mean" => post_γ_logit_mean,
      # "post_γ_logit_std_L" => post_γ_logit_std_L,

      # "post_λ_logit_mean" => post_λ_logit_mean,
      # "post_λ_logit_std_L" => post_λ_logit_std_L,
    )

    stringdata = JSON.json(posterior_params)

    open(joinpath(args[:output_dir], json_name), "w") do f
        write(f, stringdata)
    end
end

function checkpoint_guide_params(args, json_name)
    # Fetch learned params from guide
    post_E0_logit_means = get_param(guide, :E0_logit_means)
    post_E0_logit_std_L = get_param(guide, :E0_logit_std_L)

    post_βE_logit_means = get_param(guide, :βE_logit_means)
    post_βE_logit_std_Ls = get_param(guide, :βE_logit_std_Ls)

    post_βI_logit_means = get_param(guide, :βI_logit_means)
    post_βI_logit_std_Ls = get_param(guide, :βI_logit_std_Ls)

    # post_γ_logit_mean = get_param(guide, :γ_logit_mean)
    # post_γ_logit_std_L = get_param(guide, :γ_logit_std_L)

    # post_λ_logit_mean = get_param(guide, :λ_logit_mean)
    # post_λ_logit_std_L = get_param(guide, :λ_logit_std_L)

    # Save posterior params to JSON file
    posterior_params = Dict(
      "post_E0_logit_means" => post_E0_logit_means,
      "post_E0_logit_std_L" => post_E0_logit_std_L,

      "fixed_total_E0" => args[:fixed_total_E0],

      "post_βE_logit_means" => post_βE_logit_means,
      "post_βE_logit_std_Ls" => post_βE_logit_std_Ls,

      "post_βI_logit_means" => post_βI_logit_means,
      "post_βI_logit_std_Ls" => post_βI_logit_std_Ls,

      # "post_γ_logit_mean" => post_γ_logit_mean,
      # "post_γ_logit_std_L" => post_γ_logit_std_L,

      # "post_λ_logit_mean" => post_λ_logit_mean,
      # "post_λ_logit_std_L" => post_λ_logit_std_L,
    )

    stringdata = JSON.json(posterior_params)

    open(joinpath(args[:output_dir], json_name), "w") do f
        write(f, stringdata)
    end
end

# Assumes logit-normal variable
function print_param_summary(io, name, prior_mean::Float64, prior_std_L::Float64, post_mean::Array{Float64}, post_std_L::Array{Float64})
    println(io, name)
    println(io, ("prior params    ", prior_mean, prior_std_L))
    println(io, ("prior med       ", sigmoid(prior_mean)))
    println(io, ("prior med + 1std", sigmoid(prior_mean + exp(prior_std_L))))
    println(io, ("prior med - 1std", sigmoid(prior_mean - exp(prior_std_L))))
    println(io, ("post params     ", post_mean, post_std_L))
    println(io, ("post med        ", map(x -> sigmoid(x), post_mean)))
    println(io, ("post med + 1std ", map((x, y) -> sigmoid(x + exp(y)), post_mean, post_std_L)))
    println(io, ("post med - 1std ", map((x, y) -> sigmoid(x - exp(y)), post_mean, post_std_L)))
end

# Assumes logit-normal variable
function print_normalized_E0_summary(args, io, name, prior_mean::Float64, prior_std_L::Float64, post_mean::Array{Float64}, post_std_L::Array{Float64}, community_sizes::Array{Int})
    println(io, name)
    println(io, ("prior params    ", prior_mean, prior_std_L))
    println(io, ("post params     ", post_mean, post_std_L))
    println(io, ("post med        ", normalize_E0(args, community_sizes, map(x -> sigmoid(x), post_mean))))
    println(io, ("post med + 1std ", normalize_E0(args, community_sizes, map((x, y) -> sigmoid(x + exp(y)), post_mean, post_std_L))))
    println(io, ("post med - 1std ", normalize_E0(args, community_sizes, map((x, y) -> sigmoid(x - exp(y)), post_mean, post_std_L))))

    fig = Plots.bar(1:length(community_sizes), normalize_E0(args, community_sizes, map(x -> sigmoid(x), post_mean)))
    xticks!(1:length(community_sizes))
    savefig(fig, joinpath(args[:output_dir], "E0_medians.png"))
end

function output_jh_csv(args, filename_confirmed, filename_deaths, post_daily_counts, num_nodes)
    confirmed_header = "UID,iso2,iso3,code3,FIPS,Admin2,Province_State,Country_Region,Lat,Long_,Combined_Key,1/22/20,1/23/20,1/24/20,1/25/20,1/26/20,1/27/20,1/28/20,1/29/20,1/30/20,1/31/20,2/1/20,2/2/20,2/3/20,2/4/20,2/5/20,2/6/20,2/7/20,2/8/20,2/9/20,2/10/20,2/11/20,2/12/20,2/13/20,2/14/20,2/15/20,2/16/20,2/17/20,2/18/20,2/19/20,2/20/20,2/21/20,2/22/20,2/23/20,2/24/20,2/25/20,2/26/20,2/27/20,2/28/20,2/29/20,3/1/20,3/2/20,3/3/20,3/4/20,3/5/20,3/6/20,3/7/20,3/8/20,3/9/20,3/10/20,3/11/20,3/12/20,3/13/20,3/14/20,3/15/20,3/16/20,3/17/20,3/18/20,3/19/20,3/20/20,3/21/20,3/22/20,3/23/20,3/24/20,3/25/20,3/26/20,3/27/20,3/28/20,3/29/20,3/30/20,3/31/20,4/1/20,4/2/20,4/3/20,4/4/20,4/5/20,4/6/20,4/7/20,4/8/20,4/9/20,4/10/20,4/11/20,4/12/20,4/13/20,4/14/20,4/15/20,4/16/20,4/17/20,4/18/20,4/19/20,4/20/20,4/21/20,4/22/20,4/23/20,4/24/20,4/25/20,4/26/20,4/27/20,4/28/20,4/29/20,4/30/20,5/1/20,5/2/20,5/3/20,5/4/20,5/5/20,5/6/20,5/7/20,5/8/20,5/9/20,5/10/20,5/11/20,5/12/20,5/13/20,5/14/20,5/15/20,5/16/20,5/17/20,5/18/20,5/19/20,5/20/20,5/21/20,5/22/20,5/23/20,5/24/20,5/25/20,5/26/20,5/27/20,5/28/20,5/29/20,5/30/20,5/31/20,6/1/20,6/2/20,6/3/20,6/4/20,6/5/20,6/6/20,6/7/20,6/8/20,6/9/20,6/10/20,6/11/20,6/12/20,6/13/20,6/14/20,6/15/20,6/16/20,6/17/20,6/18/20,6/19/20,6/20/20,6/21/20,6/22/20,6/23/20,6/24/20,6/25/20,6/26/20,6/27/20,6/28/20,6/29/20,6/30/20,7/1/20,7/2/20,7/3/20,7/4/20,7/5/20,7/6/20,7/7/20,7/8/20,7/9/20,7/10/20,7/11/20,7/12/20,7/13/20,7/14/20,7/15/20,7/16/20,7/17/20,7/18/20,7/19/20,7/20/20,7/21/20,7/22/20,7/23/20,7/24/20,7/25/20,7/26/20,7/27/20,7/28/20,7/29/20,7/30/20,7/31/20,8/1/20,8/2/20,8/3/20,8/4/20,8/5/20,8/6/20,8/7/20,8/8/20,8/9/20,8/10/20,8/11/20,8/12/20,8/13/20,8/14/20,8/15/20,8/16/20,8/17/20,8/18/20,8/19/20,8/20/20,8/21/20,8/22/20,8/23/20,8/24/20,8/25/20,8/26/20,8/27/20,8/28/20,8/29/20,8/30/20,8/31/20,9/1/20,9/2/20,9/3/20,9/4/20,9/5/20,9/6/20,9/7/20,9/8/20,9/9/20,9/10/20,9/11/20,9/12/20,9/13/20,9/14/20,9/15/20,9/16/20,9/17/20,9/18/20,9/19/20,9/20/20,9/21/20,9/22/20,9/23/20,9/24/20,9/25/20,9/26/20,9/27/20,9/28/20,9/29/20,9/30/20,10/1/20,10/2/20,10/3/20,10/4/20,10/5/20,10/6/20,10/7/20,10/8/20,10/9/20,10/10/20,10/11/20,10/12/20,10/13/20,10/14/20,10/15/20,10/16/20,10/17/20,10/18/20,10/19/20,10/20/20,10/21/20,10/22/20,10/23/20,10/24/20,10/25/20,10/26/20,10/27/20,10/28/20,10/29/20,10/30/20,10/31/20,11/1/20,11/2/20,11/3/20,11/4/20,11/5/20,11/6/20,11/7/20,11/8/20,11/9/20,11/10/20,11/11/20,11/12/20,11/13/20,11/14/20,11/15/20,11/16/20,11/17/20,11/18/20,11/19/20,11/20/20,11/21/20,11/22/20,11/23/20,11/24/20,11/25/20,11/26/20,11/27/20,11/28/20,11/29/20,11/30/20,12/1/20,12/2/20,12/3/20,12/4/20,12/5/20,12/6/20,12/7/20,12/8/20,12/9/20,12/10/20,12/11/20,12/12/20,12/13/20,12/14/20,12/15/20,12/16/20,12/17/20,12/18/20,12/19/20,12/20/20,12/21/20,12/22/20,12/23/20,12/24/20,12/25/20,12/26/20,12/27/20,12/28/20,12/29/20,12/30/20,12/31/20,1/1/21,1/2/21,1/3/21,1/4/21,1/5/21,1/6/21,1/7/21,1/8/21,1/9/21,1/10/21,1/11/21,1/12/21"
    deaths_header =    "UID,iso2,iso3,code3,FIPS,Admin2,Province_State,Country_Region,Lat,Long_,Combined_Key,Population,1/22/20,1/23/20,1/24/20,1/25/20,1/26/20,1/27/20,1/28/20,1/29/20,1/30/20,1/31/20,2/1/20,2/2/20,2/3/20,2/4/20,2/5/20,2/6/20,2/7/20,2/8/20,2/9/20,2/10/20,2/11/20,2/12/20,2/13/20,2/14/20,2/15/20,2/16/20,2/17/20,2/18/20,2/19/20,2/20/20,2/21/20,2/22/20,2/23/20,2/24/20,2/25/20,2/26/20,2/27/20,2/28/20,2/29/20,3/1/20,3/2/20,3/3/20,3/4/20,3/5/20,3/6/20,3/7/20,3/8/20,3/9/20,3/10/20,3/11/20,3/12/20,3/13/20,3/14/20,3/15/20,3/16/20,3/17/20,3/18/20,3/19/20,3/20/20,3/21/20,3/22/20,3/23/20,3/24/20,3/25/20,3/26/20,3/27/20,3/28/20,3/29/20,3/30/20,3/31/20,4/1/20,4/2/20,4/3/20,4/4/20,4/5/20,4/6/20,4/7/20,4/8/20,4/9/20,4/10/20,4/11/20,4/12/20,4/13/20,4/14/20,4/15/20,4/16/20,4/17/20,4/18/20,4/19/20,4/20/20,4/21/20,4/22/20,4/23/20,4/24/20,4/25/20,4/26/20,4/27/20,4/28/20,4/29/20,4/30/20,5/1/20,5/2/20,5/3/20,5/4/20,5/5/20,5/6/20,5/7/20,5/8/20,5/9/20,5/10/20,5/11/20,5/12/20,5/13/20,5/14/20,5/15/20,5/16/20,5/17/20,5/18/20,5/19/20,5/20/20,5/21/20,5/22/20,5/23/20,5/24/20,5/25/20,5/26/20,5/27/20,5/28/20,5/29/20,5/30/20,5/31/20,6/1/20,6/2/20,6/3/20,6/4/20,6/5/20,6/6/20,6/7/20,6/8/20,6/9/20,6/10/20,6/11/20,6/12/20,6/13/20,6/14/20,6/15/20,6/16/20,6/17/20,6/18/20,6/19/20,6/20/20,6/21/20,6/22/20,6/23/20,6/24/20,6/25/20,6/26/20,6/27/20,6/28/20,6/29/20,6/30/20,7/1/20,7/2/20,7/3/20,7/4/20,7/5/20,7/6/20,7/7/20,7/8/20,7/9/20,7/10/20,7/11/20,7/12/20,7/13/20,7/14/20,7/15/20,7/16/20,7/17/20,7/18/20,7/19/20,7/20/20,7/21/20,7/22/20,7/23/20,7/24/20,7/25/20,7/26/20,7/27/20,7/28/20,7/29/20,7/30/20,7/31/20,8/1/20,8/2/20,8/3/20,8/4/20,8/5/20,8/6/20,8/7/20,8/8/20,8/9/20,8/10/20,8/11/20,8/12/20,8/13/20,8/14/20,8/15/20,8/16/20,8/17/20,8/18/20,8/19/20,8/20/20,8/21/20,8/22/20,8/23/20,8/24/20,8/25/20,8/26/20,8/27/20,8/28/20,8/29/20,8/30/20,8/31/20,9/1/20,9/2/20,9/3/20,9/4/20,9/5/20,9/6/20,9/7/20,9/8/20,9/9/20,9/10/20,9/11/20,9/12/20,9/13/20,9/14/20,9/15/20,9/16/20,9/17/20,9/18/20,9/19/20,9/20/20,9/21/20,9/22/20,9/23/20,9/24/20,9/25/20,9/26/20,9/27/20,9/28/20,9/29/20,9/30/20,10/1/20,10/2/20,10/3/20,10/4/20,10/5/20,10/6/20,10/7/20,10/8/20,10/9/20,10/10/20,10/11/20,10/12/20,10/13/20,10/14/20,10/15/20,10/16/20,10/17/20,10/18/20,10/19/20,10/20/20,10/21/20,10/22/20,10/23/20,10/24/20,10/25/20,10/26/20,10/27/20,10/28/20,10/29/20,10/30/20,10/31/20,11/1/20,11/2/20,11/3/20,11/4/20,11/5/20,11/6/20,11/7/20,11/8/20,11/9/20,11/10/20,11/11/20,11/12/20,11/13/20,11/14/20,11/15/20,11/16/20,11/17/20,11/18/20,11/19/20,11/20/20,11/21/20,11/22/20,11/23/20,11/24/20,11/25/20,11/26/20,11/27/20,11/28/20,11/29/20,11/30/20,12/1/20,12/2/20,12/3/20,12/4/20,12/5/20,12/6/20,12/7/20,12/8/20,12/9/20,12/10/20,12/11/20,12/12/20,12/13/20,12/14/20,12/15/20,12/16/20,12/17/20,12/18/20,12/19/20,12/20/20,12/21/20,12/22/20,12/23/20,12/24/20,12/25/20,12/26/20,12/27/20,12/28/20,12/29/20,12/30/20,12/31/20,1/1/21,1/2/21,1/3/21,1/4/21,1/5/21,1/6/21,1/7/21,1/8/21,1/9/21,1/10/21,1/11/21,1/12/21"

    open(filename_confirmed, "w") do f
        println(f, confirmed_header)
        for i in 1:size(post_daily_counts)[1]
            cumulative_inf = [x.EI for x in post_daily_counts[i]]
            cumulative_inf ./= args[:undercount_factor] / args[:county_pop] * num_nodes
            cumsum!(cumulative_inf, cumulative_inf)
            line_start = repeat("$i,", 5) * args[:jhu_county_name] * "," * args[:jhu_state_name] * "," * repeat("$i,", 4)
            line_end = repeat("0,", args[:start_day] - 1) * join(cumulative_inf, ",") * repeat(",$(cumulative_inf[end])", 356 - args[:end_day])
            println(f, line_start * line_end)
        end
    end

    open(filename_deaths, "w") do f
        println(f, deaths_header)
        for i in 1:size(post_daily_counts)[1]
            cumulative_dead = [x.R * args[:mortality_rate] for x in post_daily_counts[i]]
            # See usage above;
            cumulative_dead ./= args[:death_undercount_factor] / args[:county_pop] * num_nodes
            line_start = repeat("$i,", 5) * args[:jhu_county_name] * "," * args[:jhu_state_name] * "," * repeat("$i,", 5)
            line_end = repeat("0,", args[:start_day] - 1) * join(cumulative_dead, ",") * repeat(",$(cumulative_dead[end])", 356 - args[:end_day])
            println(f, line_start * line_end)
        end
    end


end
