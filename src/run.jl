function run_inference(args, io)
    println(io, "Begin run_inference")

    println(io, args)

    # Load graph instance
    rng  = reset_seed(args[:random_seed])

    if isnothing(args[:validation_type])
        g = GraphSEIR.load_graph(args[:path_node_attributes], args[:path_edge_attributes], scale=args[:scaling_factor])

        if args[:time_varying]
            # First edge file loaded already; now load the rest
            time_varying_edges = [GraphSEIR.load_edgefile(get_prop(g, :vlist), e, scale=args[:scaling_factor]) for e in args[:time_varying_edge_files][2:end]]
        else
            time_varying_edges = []
        end

        if !isnothing(args[:pct_downsample_graph])
            println(io, "BEFORE REMOVING VERTICES: num vertices: $(nv(g)), nnz in weight_mat: $(nnz(get_prop(g, :weight_matrix)))")
            indices_to_remove = 1:Int(floor(args[:pct_downsample_graph] * nv(g)))
            for i in indices_to_remove
                rem_vertex!(g, i)
            end
            
            edges = JSON.parsefile(args[:path_edge_attributes])
            vertices = JSON.parsefile(args[:path_node_attributes])
            vlist = sort(unique(vertex["id"] for vertex in vertices))
            for edge in edges
                edge["source"] = findfirst(isequal(edge["source"]), vlist) - 1
                edge["target"] = findfirst(isequal(edge["target"]), vlist) - 1
            end
            filt_edges = [x for x in edges if (((x["source"] + 1) ∉ indices_to_remove) && ((x["target"] + 1) ∉ indices_to_remove))]
            filt_vlist = [v for (idx, v) in enumerate(vlist) if (idx ∉ indices_to_remove)]
            for edge in filt_edges
                edge["source"] = findfirst(isequal(edge["source"]), filt_vlist) - 1
                edge["target"] = findfirst(isequal(edge["target"]), filt_vlist) - 1
            end

            I = [edge["source"] + 1 for edge in filt_edges]
            J = [edge["target"] + 1 for edge in filt_edges]
            V = [Float64(edge["weight"]) for edge in filt_edges] .* args[:scaling_factor]
            weight_mat = sparse(vcat(I, J), vcat(J, I), vcat(V, V), nv(g), nv(g), +)
            set_prop!(g, :weight_matrix, weight_mat)
            # NOTE - other properties on g may be wrong, such as:
            # - num_communities (if all edges in a community got removed)
            # - community_sizes (definitely wrong, but we set this manually below)
            println(io, "AFTER REMOVING VERTICES: num vertices: $(nv(g)), nnz in weight_mat: $(nnz(get_prop(g, :weight_matrix)))")
        end

    elseif args[:validation_type] in ["complete", "komplett"]
        n = 100
        g = make_komplett_graph(n)
    else # TODO - cycle, regular
        throw("unimplemented")
    end

    num_communities = get_prop(g, :num_communities)
    community_sizes = [size(collect(filter_vertices(g, :community, c)))[1] for c in 1:num_communities]
    set_prop!(g, :community_sizes, community_sizes)

    if args[:tiny]
        args[:iterations] = 2
        args[:samples_per_iter] = 4
    end

    if isnothing(args[:validation_type])
        obs_traj = load_prep_data(args, nv(g))
    elseif args[:validation_type] in ["complete", "komplett"]
        throw("unimplemented")
    else # TODO - cycle, regular
        throw("unimplemented")
    end

    println(io, args)

    # Plots.plot(obs_traj, label=["I" "D"])
    println(io, (count_S(g), count_E(g), count_I(g)))

    if args[:inf_strat] == "bbvi"
        # BBVI
        elbos = @time infer_params(args, io, g, obs_traj, rng,
                                iters=args[:iterations],
                                samples_per_iter=args[:samples_per_iter],
                                lr=args[:lr],
                                decay=args[:decay],
                                obs_noise_std=args[:obs_noise_std],
                                time_varying_edges=time_varying_edges)
        savefig(Plots.plot(elbos, left_margin=50Plots.px), joinpath(args[:output_dir], "elbos.png"))

        output_guide_params(args, io, community_sizes, size(obs_traj)[1], "posterior_params.json", "Final----------------------------------")

    elseif args[:inf_strat] == "is"
        # Importance Sampling (likelihood weighting)
        (traces, log_normalized_weights) = @time infer_params(args, io, g, obs_traj, rng,
                                iters=args[:iterations],
                                samples_per_iter=args[:samples_per_iter],
                                lr=args[:lr],
                                decay=args[:decay],
                                obs_noise_std=args[:obs_noise_std],
                                time_varying_edges=time_varying_edges)

        println(size(traces))
        println(size(log_normalized_weights))

        # Plotting
        obs_traj = load_prep_data(args, nv(g))
        num_Ts = size(obs_traj)[1]
        post_daily_counts = []
        prior_daily_counts = []
        posterior = Categorical(map(exp, log_normalized_weights))

        for i in 1:args[:num_traj]
            if (i % 5 == 0)
                println(io, "traj $(i)")
            end
            idx = rand(posterior)
            push!(post_daily_counts, traces[idx].retval)
            push!(prior_daily_counts, sample_from_prior(args, io, g, rng, num_Ts, args[:obs_noise_std], []))
        end

        jldopen(joinpath(args[:output_dir], "compare_post_daily_counts.num_traj=$(args[:num_traj]).jld"), "w") do file
            write(file, "post_daily_counts", post_daily_counts)
            write(file, "prior_daily_counts", prior_daily_counts)
        end

        savefig(plot_SEIR(post_daily_counts),
                joinpath(args[:output_dir], "posterior_seir_curve.num_traj=$(args[:num_traj]).png"))

        savefig(plot_SEIR(prior_daily_counts),
                joinpath(args[:output_dir], "prior_seir_curve.num_traj=$(args[:num_traj]).png"))

        savefig(plot_inf(prior_daily_counts, post_daily_counts, obs_traj[:,1]),
                joinpath(args[:output_dir], "cumulative_inf_comparison.num_traj=$(args[:num_traj]).png"))

        output_jh_csv(args,joinpath(args[:output_dir], "generated_trajectories_confirmed.csv"),
                           joinpath(args[:output_dir], "generated_trajectories_deaths.csv"),post_daily_counts,nv(g))


    elseif args[:inf_strat] == "mh"
        # Lightweight Metropolis Hastings
        (all_traces, all_accept_reject_outcomes, all_selected_idx, all_log_weights) = @time infer_params(args, io, g, obs_traj, rng,
                                iters=args[:iterations],
                                samples_per_iter=args[:samples_per_iter],
                                lr=args[:lr],
                                decay=args[:decay],
                                obs_noise_std=args[:obs_noise_std],
                                time_varying_edges=time_varying_edges)    

        println(io, ("all_accept_reject_outcomes", all_accept_reject_outcomes))
        println(io, ("all_selected_idx", all_selected_idx))
        println(io, ("all_log_weights", all_log_weights))

        # Plotting
        ## Select final N samples from each chain
        @assert args[:num_traj] % args[:num_chains] == 0
        samples_per_chain = Int(args[:num_traj] / args[:num_chains])
        println(io, ("samples_per_chain", samples_per_chain))
        post_daily_counts = []
        for chain in all_traces
            append!(post_daily_counts, map(x -> x.retval, chain[end - samples_per_chain + 1:end]))
        end
        println(io, ("post_daily_counts length", length(post_daily_counts)))
        obs_traj = load_prep_data(args, nv(g))
        num_Ts = size(obs_traj)[1]
        prior_daily_counts = []

        for _ in 1:args[:num_traj]
            push!(prior_daily_counts, sample_from_prior(args, io, g, rng, num_Ts, args[:obs_noise_std], []))
        end

        jldopen(joinpath(args[:output_dir], "compare_post_daily_counts.num_traj=$(args[:num_traj]).jld"), "w") do file
            write(file, "post_daily_counts", post_daily_counts)
            write(file, "prior_daily_counts", prior_daily_counts)
        end

        savefig(plot_SEIR(post_daily_counts),
                joinpath(args[:output_dir], "posterior_seir_curve.num_traj=$(args[:num_traj]).png"))

        savefig(plot_SEIR(prior_daily_counts),
                joinpath(args[:output_dir], "prior_seir_curve.num_traj=$(args[:num_traj]).png"))

        savefig(plot_inf(prior_daily_counts, post_daily_counts, obs_traj[:,1]),
                joinpath(args[:output_dir], "cumulative_inf_comparison.num_traj=$(args[:num_traj]).png"))

        output_jh_csv(args,joinpath(args[:output_dir], "generated_trajectories_confirmed.csv"),
                           joinpath(args[:output_dir], "generated_trajectories_deaths.csv"),post_daily_counts,nv(g))
    end
end

function run_compare(args, zero_variance=false, show_prior=true)
    # Plot 2 trajectories: original observations and network model
    @assert isfile(args[:json_results])
    json_results = JSON.parsefile(args[:json_results])
    # TODO - run_compare no longer stands alone for re-generating plots from a previous folder.
    # args[:output_dir] = dirname(args[:json_results])
    # @assert isdir(args[:output_dir])

    filesuffix = ""
    if (zero_variance)
        filesuffix = ".zero_var"
    end

    if (show_prior)
        mode_text = "compare"
    else
        mode_text = "generate"
    end

    # TODO - redirect stderr to this log
    open(joinpath(args[:output_dir], mode_text * "_log$(filesuffix).txt"), "w") do io
        println(io, "Begin run_" * mode_text)

        println(io, args)

        println(io, json_results)

        init_param!(guide, :E0_logit_means, convert(Array{Float64}, json_results["post_E0_logit_means"]))
        init_param!(guide, :E0_logit_std_L, json_results["post_E0_logit_std_L"])

        args[:fixed_total_E0] = json_results["fixed_total_E0"]

        init_param!(guide, :βE_logit_means, convert(Array{Float64}, json_results["post_βE_logit_means"]))
        init_param!(guide, :βE_logit_std_Ls, convert(Array{Float64}, json_results["post_βE_logit_std_Ls"]))

        init_param!(guide, :βI_logit_means, convert(Array{Float64}, json_results["post_βI_logit_means"]))
        init_param!(guide, :βI_logit_std_Ls, convert(Array{Float64}, json_results["post_βI_logit_std_Ls"]))

        # init_param!(guide, :γ_logit_mean, json_results["post_γ_logit_mean"])
        # init_param!(guide, :γ_logit_std_L, json_results["post_γ_logit_std_L"])

        # init_param!(guide, :λ_logit_mean, json_results["post_λ_logit_mean"])
        # init_param!(guide, :λ_logit_std_L, json_results["post_λ_logit_std_L"])

        if (zero_variance)
            println(io, "ZERO VARIANCE MODE")
            init_param!(guide, :E0_logit_std_L, -Inf)
            init_param!(guide, :βE_logit_std_Ls, -Inf * ones(length(get_param(guide, :βE_logit_std_Ls))))
            init_param!(guide, :βI_logit_std_Ls, -Inf * ones(length(get_param(guide, :βI_logit_std_Ls))))
            # init_param!(guide, :γ_logit_std_L, -Inf)
            # init_param!(guide, :λ_logit_std_L, -Inf)
            args[:obs_noise_std] = 0.0
        end

        rng  = reset_seed(args[:random_seed])
        g = GraphSEIR.load_graph(args[:path_node_attributes], args[:path_edge_attributes], scale=args[:scaling_factor])

        if args[:time_varying]
            # First edge file loaded already; now load the rest
            time_varying_edges = [GraphSEIR.load_edgefile(get_prop(g, :vlist), e, scale=args[:scaling_factor]) for e in args[:time_varying_edge_files][2:end]]
        else
            time_varying_edges = []
        end

        obs_traj = load_prep_data(args, nv(g))
        if (show_prior)
            num_T = size(obs_traj)[1]
        else
            num_T = args[:end_day] - args[:start_day] + 1
        end

        if args[:output_terms]
            observations = choicemap()
            num_Ts = size(obs_traj)[1]
            for i in 1:num_Ts
                observations[(i, :I)] = obs_traj[i,1]
                if args[:cond_on_deaths]
                    observations[(i, :D)] = obs_traj[i,2]
                end
            end

            model_args =  (args, g, num_Ts, rng, args[:obs_noise_std], time_varying_edges)
            z_guide = get_choices(Gen.simulate(guide, (args, g,)))
            trace, log_prior = Gen.generate(pandemic_model, model_args, z_guide)
            log_joint, retval = Gen.assess(pandemic_model, model_args, Gen.merge(z_guide, observations))

            println(io, "---------z sampled from guide-----------")
            println(io, "log_prior,log_joint,log_likelihood")
            println(io, (log_prior, log_joint, log_joint - log_prior))
            # If inference succeeds, the "golden test" would be that the variational dist. is close to the true posterior.
            # (e.g. measured using KL)
            # As a quick check that a few samples from our variational post. should have higher likelihood and should have higher prob under true posterior
            # and also have higher prob. under likelihood when compared to the prior.
            
            # Specifically:
            # We can't measure the true posterior p(z|x), but we CAN measure p(x, z), and we know p(x) is same for both posterior and prior (so the joint is proportional)
            # Thus: log_joint using posterior should be higher.
            # Note: failing this "test" shows that inference failed. But if we succeed, we might still have
            # a variational posterior that collapsed to one mode of the true posterior, or always samples from the highest point of the true posterior, etc.
            # In other words - this quick check does NOT actually measure the KL, it's just a sanity check.

            # To make this sanity check slightly better, we could compare KL(q(z) || p(z|x)) with KL(p(z) || p(z|x))
            # Go through the same derivation as we use for the ELBO, and we see we have almost computed all the required terms

            z_prior = get_choices(Gen.simulate(pandemic_model, model_args))
            merged_choices = choicemap()
            for (k, v) in Gen.get_values_shallow(observations)
                merged_choices[k] = v
            end
            for (k, v) in Gen.get_values_shallow(z_prior)
                if !Gen.has_value(merged_choices, k)
                    merged_choices[k] = v
                end
            end


            trace, log_prior = Gen.generate(pandemic_model, model_args, z_prior)
            log_joint, retval = Gen.assess(pandemic_model, model_args, merged_choices)

            println(io, "---------z sampled from prior-----------")
            println(io, "log_prior,log_joint,log_likelihood")
            println(io, (log_prior, log_joint, log_joint - log_prior))
            # func(x)
            #   z ~ p(.)
            #   return p(x,z), p(x | z) 
            #
            #   assess => p(x,z) -- joint
            #      p(x | z) = p(x,z) / p(z)
            #   generate => p(x | z) -- Likelihood
            #   p(z) -- prior
            #       p(z) = p(x,z) / p(x | z)
            #       log(p(z)) = log(p(x,z)) - log(p(x | z))
        end

        post_daily_counts = []
        prior_daily_counts = []
        for i in 1:args[:num_traj]
            if (i % 5 == 0)
                println(io, "traj $(i)")
            end
            push!(post_daily_counts, sample_from_posterior(args, io, g, rng, num_T, args[:obs_noise_std], time_varying_edges))
            if (show_prior)
                push!(prior_daily_counts, sample_from_prior(args, io, g, rng, num_T, args[:obs_noise_std], time_varying_edges))
            end
        end

        jldopen(joinpath(args[:output_dir], "compare_post_daily_counts$(filesuffix).num_traj=$(args[:num_traj]).jld"), "w") do file
            write(file, "post_daily_counts", post_daily_counts)
            if (show_prior)
                write(file, "prior_daily_counts", prior_daily_counts)
            end
        end

        savefig(plot_SEIR(post_daily_counts),
                joinpath(args[:output_dir], "posterior_seir_curve$(filesuffix).num_traj=$(args[:num_traj]).png"))

        if (show_prior)
            savefig(plot_SEIR(prior_daily_counts),
                    joinpath(args[:output_dir], "prior_seir_curve$(filesuffix).num_traj=$(args[:num_traj]).png"))

            savefig(plot_inf(prior_daily_counts, post_daily_counts, obs_traj[:,1]),
                    joinpath(args[:output_dir], "cumulative_inf_comparison$(filesuffix).num_traj=$(args[:num_traj]).png"))
        end

        output_jh_csv(args,joinpath(args[:output_dir], "generated_trajectories_confirmed$(filesuffix).csv"),
                           joinpath(args[:output_dir], "generated_trajectories_deaths$(filesuffix).csv"),post_daily_counts,nv(g))

    end
end
