const prior_βI_logit_mean = -5.05 # logit(0.018)
# const args[:prior_gamma_logit_mean] = logit(0.048)  # numbers taken from Lucas
# const args[:prior_lambda_logit_mean] = logit(0.064)

# Updated 3-18-21 using papers
# const args[:prior_gamma_logit_mean] = logit(6.7e-2)  # numbers taken from Lucas
# const args[:prior_lambda_logit_mean] = logit(1.1e-1)



sigmoid(x) = StatsFuns.logistic(x)

function normalize_E0(args, community_sizes, E0_pcts)
    n_exposed_total = sum(community_sizes .* E0_pcts)
    rescale_factor = args[:fixed_total_E0] * sum(community_sizes) / n_exposed_total
    return E0_pcts .* rescale_factor
end

function get_noise(obs_noise_std, g, day, cumulative_inf, noise_fn)
    # Scale std linearly with # nodes
    gs = nv(g)
    if noise_fn == "day_scaling"
        noise = obs_noise_std * gs * (sqrt(day) + 2 / gs)    # 0bs_noise_std 0.00025
    elseif noise_fn == "mag_scaling" 
        noise = max(cumulative_inf[day] * obs_noise_std, 2) # obs_noise_std  1/150-> 0.0066
    elseif noise_fn == "constant"
        noise = obs_noise_std * gs
    elseif noise_fn == "step"
        if day <= 30
            noise = obs_noise_std / 2 * gs
        else
            noise = obs_noise_std * gs
        end
    else
        ArgumentError("Unknown Noise Function")
    end
    # The noise level on day 0 might be too small; so we "start on day 2"
    
    return noise
end


function run_pandemic(args, g, days, rng, inf_tree, βE::Vector{Float64}, βI::Vector{Float64}, γ, λ, time_varying_edges)
    # Fit spline for beta_E and beta_I
    if args[:knots] > 1
        knot_days = push!(map(x -> convert(Int, floor(x)), collect(1:days / (args[:knots] - 1):days)), days)
        spline_βE = Spline1D(knot_days, βE, k=1) # NOTE - should we fit in log space or not?
        spline_βI = Spline1D(knot_days, βI, k=1)
    else
        spline_βE(day) = βE[end]
        spline_βI(day) = βI[end]
    end
    tracker = CompartmentalTracker([0]) # 0 is a magic community number; tracks global stats
    for day in 1:days
        if args[:time_varying] && day % 7 == 0
            idx = Int(floor(day / 7))
            idx = min(length(time_varying_edges), idx)
            GraphSEIR.swap_edges!(g, time_varying_edges[idx]...)
        end
        today_βE = spline_βE(day)
        today_βI = spline_βI(day)
        transition!(g=g, params=GlobalParameters(today_βE, today_βI, γ, λ), time_index=day, tracker=tracker, rng=rng)
    end
    return tracker[0]
end


@gen function pandemic_model(args, g, days, rng, obs_noise_std, time_varying_edges)
    # We'll modify the graph inplace, so we operate on a copy
    g = deepcopy(g)

    num_communities = get_prop(g, :num_communities)
    community_sizes = get_prop(g, :community_sizes) 

    # Set prior on params
    # NOTE - std set such that adding one std from the mean corresponds to adding 1 to the param

    E0_pcts = zeros(num_communities)
    for i in 1:num_communities
        p = @trace(normal(args[:prior_E0_logit_mean], exp(args[:prior_E0_logit_std_L])), (i, :E0_logit))
        E0_pcts[i] = sigmoid(p)
    end


    # If desired, rescale per-community exposures
    if args[:fixed_total_E0] != nothing
        E0_pcts = normalize_E0(args, community_sizes, E0_pcts)
    end

    inf_tree = GraphSEIR.initialize_with!(g, (g, v) -> expose_by_comm(g, v, E0_pcts), rng)

    βE = Vector{Float64}()
    βI = Vector{Float64}()
    for i in 1:args[:knots]
        b1 = @trace(normal(args[:prior_betaE_logit_mean], exp(args[:prior_beta_logit_std_L])), (i, :βE_logit))
        push!(βE, sigmoid(b1))

        b2 = @trace(normal(prior_βI_logit_mean, exp(args[:prior_beta_logit_std_L])), (i, :βI_logit))
        push!(βI, sigmoid(b2))
    end

    # γ_logit = @trace(normal(args[:prior_gamma_logit_mean], exp(args[:prior_gamma_lambda_logit_std_L])), :γ_logit)
    γ = sigmoid(args[:prior_gamma_logit_mean])

    # λ_logit = @trace(normal(args[:prior_lambda_logit_mean], exp(args[:prior_gamma_lambda_logit_std_L])), :λ_logit)
    λ = sigmoid(args[:prior_lambda_logit_mean])

    daily_counts = run_pandemic(args, g, days, rng, inf_tree, βE, βI, γ, λ, time_varying_edges)
    cumulative_inf = [x.EI for x in daily_counts]
    cumsum!(cumulative_inf, cumulative_inf)
    cumulative_death = [x.R for x in daily_counts] * args[:mortality_rate]

    # Compute Likelihood
    for day in 1:days
        # Scale std linearly with # nodes
        # true_counts = @trace(my_fake_dist(cumulative_inf[day]))
        noise = get_noise(obs_noise_std, g, day, cumulative_inf, args[:noise_fn])
        @trace(normal(cumulative_inf[day], noise), (day, :I))
        if (args[:cond_on_deaths])
            death_noise = get_noise(obs_noise_std, g, day, cumulative_death, args[:noise_fn])
            @trace(normal(cumulative_death[day], death_noise), (day, :D))
        end 
    end

    return daily_counts
end;

@gen function guide(args, g)
    @param E0_logit_means::Array{Float64,1}
    @param E0_logit_std_L::Float64

    @param βE_logit_means::Array{Float64,1}
    @param βE_logit_std_Ls::Array{Float64,1}

    @param βI_logit_means::Array{Float64,1}
    @param βI_logit_std_Ls::Array{Float64,1}

    # @param γ_logit_mean::Float64
    # @param γ_logit_std_L::Float64

    # @param λ_logit_mean::Float64
    # @param λ_logit_std_L::Float64

    for i in 1:get_prop(g, :num_communities)
        @trace(normal(E0_logit_means[i], exp(E0_logit_std_L)), (i, :E0_logit))
    end

    for i in 1:args[:knots]
        @trace(normal(βE_logit_means[i], exp(βE_logit_std_Ls[i])), (i, :βE_logit))
        @trace(normal(βI_logit_means[i], exp(βI_logit_std_Ls[i])), (i, :βI_logit))
    end

    # @trace(normal(γ_logit_mean, exp(γ_logit_std_L)), :γ_logit)
    # @trace(normal(λ_logit_mean, exp(λ_logit_std_L)), :λ_logit)
end


# black box, uses score function estimator
function single_sample_gradient_estimate!(
    var_model::GenerativeFunction, var_model_args::Tuple,
    model::GenerativeFunction, model_args::Tuple, observations::ChoiceMap,
    scale_factor=1.)

    # sample from variational approximation
    trace = Gen.simulate(var_model, var_model_args)

    # compute learning signal
    constraints = Gen.merge(observations, get_choices(trace))
    (model_log_weight, _) = Gen.assess(model, model_args, constraints)
    log_weight = model_log_weight - get_score(trace)

    # accumulate the weighted gradient
    # accumulate_param_gradients!(trace, nothing, log_weight * scale_factor)

    # unbiased estimate of objective function, and trace
    (log_weight, trace)
end


function custom_black_box_vi!(
    args,
    io,
    model::GenerativeFunction, model_args::Tuple,
    observations::ChoiceMap,
    var_model::GenerativeFunction, var_model_args::Tuple,
    update::ParamUpdate;
    iters=1000, samples_per_iter=100, verbose=false)

    function sample_chunk!(traces, log_weights, num_samples, per_thread, i)
        for j = 1:per_thread
            sample = (i - 1) * per_thread + j
            traces[sample] = simulate(var_model, var_model_args)
            constraints = merge(observations, get_choices(traces[sample]))
            model_weight, = assess(model, model_args, constraints)
            log_weights[sample] = model_weight - get_score(traces[sample])
        end
    end

    println("Running Custom BBVI")
    elbo_history = Vector{Float64}(undef, iters)
    traces = Vector{Gen.DynamicDSLTrace}(undef, samples_per_iter)
    elbos = Vector{Float64}(undef, samples_per_iter)

    @assert samples_per_iter % 4 == 0

    for iter = 1:iters
    # compute gradient estimate and objective function estimate
        per_thread = Int(samples_per_iter / 4)
        if args[:multithreaded]
            @threads for i = 1:4
                sample_chunk!(traces, elbos, samples_per_iter, per_thread, i)
            end
        else
            for i = 1:4
                sample_chunk!(traces, elbos, samples_per_iter, per_thread, i)
            end
        end
    
    # Substracting this baseline reduces the variance by "mean-shifting" the samples, 
    # while preserving the expected value due to the reinforce property
        baseline = mean(elbos)
        for i = 1:samples_per_iter
            accumulate_param_gradients!(traces[i], nothing, (elbos[i] - baseline) / samples_per_iter)
        end
        elbo_history[iter] = baseline

        verbose && println("iter $iter; average elbo: $baseline")

        apply!(update)

        output_guide_params(args, io, get_prop(model_args[2], :community_sizes), model_args[3],
                        args[:json_results_checkpoint_name], "Iter:" * string(iter) * "----------------------------------")

        if iter % 10 == 1
            name = string(iter) * "_" * args[:json_results_checkpoint_name]
            output_guide_params(args, io, get_prop(model_args[2], :community_sizes), model_args[3],
          name, "Iter:" * string(iter) * "----------------------------------")
        end
    end

    (elbo_history[end], elbo_history)
end

function metropolis_hastings(
        trace, selection::Gen.Selection;
        check=false, observations=Gen.EmptyChoiceMap())
    args = Gen.get_args(trace)
    argdiffs = map((_) -> Gen.NoChange(), args)
    (new_trace, log_weight) = Gen.regenerate(trace, args, argdiffs, selection)
    check && Gen.check_observations(Gen.get_choices(new_trace), observations)
    if log(rand()) < log_weight
        # accept
        return (new_trace, log_weight, true)
    else
        # reject
        return (trace, log_weight, false)
    end
end

function infer_params(args, io, g, obs_traj, rng;
        iters, samples_per_iter, lr,
        decay, num_samples=missing, obs_noise_std, time_varying_edges)

    observations = choicemap()
    num_Ts = size(obs_traj)[1]
    for i in 1:num_Ts
        observations[(i, :I)] = obs_traj[i,1]
        if args[:cond_on_deaths]
            observations[(i, :D)] = obs_traj[i,2]
        end
    end


    init_param!(guide, :E0_logit_means, args[:prior_E0_logit_mean] .* ones(get_prop(g, :num_communities)))
    init_param!(guide, :E0_logit_std_L, args[:prior_E0_logit_std_L])

    init_param!(guide, :βE_logit_means, args[:prior_betaE_logit_mean] .* ones(args[:knots]))
    init_param!(guide, :βE_logit_std_Ls, args[:prior_beta_logit_std_L] .* ones(args[:knots]))

    init_param!(guide, :βI_logit_means, prior_βI_logit_mean .* ones(args[:knots]))
    init_param!(guide, :βI_logit_std_Ls, args[:prior_beta_logit_std_L] .* ones(args[:knots]))

    # init_param!(guide, :γ_logit_mean, args[:prior_gamma_logit_mean])
    # init_param!(guide, :γ_logit_std_L, args[:prior_gamma_lambda_logit_std_L])

    # init_param!(guide, :λ_logit_mean, args[:prior_lambda_logit_mean])
    # init_param!(guide, :λ_logit_std_L, args[:prior_gamma_lambda_logit_std_L])

    println(io, "lr: $(lr)")
    println(io, "decay: $(decay)")
    update = ParamUpdate(GradientDescent(lr, decay), guide)

    model_args = (args, g, num_Ts, rng, obs_noise_std, time_varying_edges)
    println(io, "graph: $(model_args[2])")
    println(io, "num_Ts: $(model_args[3])")
    println(io, "obs_noise_std: $(model_args[5])")
    println(io, "time_varying_edges: $(length(model_args[6]))")
    println(io, "samples_per_iter: $(samples_per_iter)")
    println(io, "num_samples: $(num_samples)")
    println(io, "iters: $(iters)")
    println(io, "noise inf values (rel obs traj): $([get_noise(obs_noise_std, g, x, obs_traj[:,1], args[:noise_fn]) for x in 1:num_Ts])")
    println(io, "inf values: $(obs_traj[:,1])")
    println(io, "noise death values (rel obs traj): $([get_noise(obs_noise_std, g, x, obs_traj[:,2], args[:noise_fn]) for x in 1:num_Ts])")
    println(io, "death values: $(obs_traj[:,2])")

    if args[:inf_strat] == "bbvi"
        if (num_samples === missing)
            _, elbos = custom_black_box_vi!(
                args,
                io,
                pandemic_model,
                model_args,
                observations, guide, (args, g,), update, iters=iters, samples_per_iter=samples_per_iter, verbose=true);
        else
            _, traces, elbos = Gen.black_box_vimco!(
                pandemic_model,
                model_args,
                observations, guide, (args, g,), update, num_samples, iters=iters, samples_per_iter=samples_per_iter, verbose=true);
        end
        return elbos
    elseif args[:inf_strat] == "is"
        (traces, log_norm_weights, _) = importance_sampling(pandemic_model, model_args, observations, args[:samples_per_iter], true)
        return traces, log_norm_weights
     elseif args[:inf_strat] == "mh"
        # TODO:
        # - consider use the built-in function: get_values_shallow or address_set. Would need to be sure to throw out the 
        #   data observations (we don't want to resample those)
        #   https://www.gen.dev/v0.2/ref/choice_maps/#Gen.address_set
        # - ensure that the selection is done correctly - we need to see at least 1 accept.
        #   At each step, just print the entire trace, see what was selected and whether it changed
        # first generate a trace
        @assert args[:samples_per_iter] % args[:num_chains] == 0
        chain_length = args[:samples_per_iter] / args[:num_chains]

        all_traces = []
        all_accept_reject_outcomes = []
        all_selected_idx = []
        all_log_weights = []

        # Manually add all symbols that may be resampled/moved
        all_symbols = []
        for c in 1:get_prop(g, :num_communities)
            push!(all_symbols, (c, :E0_logit))
        end
        for k in 1:args[:knots]
            push!(all_symbols, (k, :βE_logit))
            push!(all_symbols, (k, :βI_logit))
        end

        # uniform distribution over symbols
        categorical = Categorical(ones(length(all_symbols)) / length(all_symbols))

        for _ in 1:args[:num_chains]
            current_chain_traces = []
            current_accepts = 0
            current_selected_idx = []
            current_log_weights = []
            trace, _ = Gen.generate(pandemic_model, (args, g, num_Ts, rng, obs_noise_std, time_varying_edges), observations)
            for _ in 1:chain_length
                # uniform random choice of variable that will be moved
                idx = rand(categorical)
                selection = Gen.select(all_symbols[idx])
                
                # move (maybe)
                (new_trace, log_weight, selected) = metropolis_hastings(trace, selection)
                current_accepts += Int(selected)
                
                # println("\n\n\nSELECTION:")
                # println(selection)
                # println("BEFORE:")
                # println(get_choices(trace))
                # println("AFTER:")
                # println(get_choices(new_trace))
                # println(("log_weight", log_weight))
                # println(selected ? "YES SELECT" : "NO SELECT")

                trace = new_trace
                push!(current_chain_traces, trace)
                push!(current_selected_idx, idx)
                push!(current_log_weights, log_weight)
            end
            push!(all_traces, current_chain_traces)
            push!(all_accept_reject_outcomes, current_accepts)
            push!(all_selected_idx, current_selected_idx)
            push!(all_log_weights, current_log_weights)
        end

        return all_traces, all_accept_reject_outcomes, all_selected_idx, all_log_weights
    end
end

function sample_from_posterior(args, io, g, rng, num_Ts, obs_noise_std, time_varying_edges)
    trace = Gen.simulate(guide, (args, g,))
    constraints = get_choices(trace)
    println(io, "POSTERIOR CONSTRAINTS - having chosen model params from the guide, what disease params did we sample?")
    println(io, constraints)
    trace, _ =  Gen.generate(pandemic_model, (args, g, num_Ts, rng, obs_noise_std, time_varying_edges), constraints)
    println(io, "POSTERIOR TRACE - after running the model using the sampled disease params, what did we get?")
    println(io, constraints)
    daily_counts = trace.retval
    return daily_counts
end

function sample_from_prior(args, io, g, rng, num_Ts, obs_noise_std, time_varying_edges)
    trace, _ =  Gen.generate(pandemic_model, (args, g, num_Ts, rng, obs_noise_std, time_varying_edges), choicemap())
    daily_counts = trace.retval
    return daily_counts
end
