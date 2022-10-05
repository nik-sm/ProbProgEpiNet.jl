function get_data(args, cumulative_counts)
    println(args[:io], "size cumulative_counts: ", size(cumulative_counts))
    inf1 = cumulative_counts[2:end]
    inf2 = cumulative_counts[1:end - 1]
    per_day_counts = inf1 - inf2
    return cumulative_counts, per_day_counts
end

function load_prep_data(args, num_nodes::Int)
    println(args[:io], args[:real_deaths])
    println(args[:io], args[:real_confirmed])
    df_deaths = read_csv(args[:real_deaths])
    df_infected = read_csv(args[:real_confirmed])

    county = args[:jhu_county_name] # Suffolk
    state = args[:jhu_state_name]  # Mass
    query_string = "Admin2 == \"$(county)\" && Province_State == \"$(state)\""
    q = Meta.parse(query_string)

    # NOTE - data begins on columns 12 and 13
    county_deaths_cum = map(x -> convert(Float64, x), Array(iloc(query(read_csv(args[:real_deaths]), q))[1,13:size(df_deaths)[2]]))
    county_infected_cum = map(x -> convert(Float64, x), Array(iloc(query(read_csv(args[:real_confirmed]), q))[1,12:size(df_infected)[2]]))

    # To Do : Remove 200 and use end_day by solving for undercount and start day simultaniously using 
    # https://jamboard.google.com/d/1wpRHDPpfBYG74GdiDm9-xUJtiEAfWN72HI9J7JdiIvc/edit?usp=sharing
    actual_final_inf_frac = county_infected_cum[200] / args[:county_pop]
    args[:undercount_factor] = args[:final_inf_saturation_frac] / actual_final_inf_frac

    # Note the quesitonable assumption in the death_undercount_factor is that the rate at which people enter R is the same
    # regardless of if people die or recover.  One way to think of this, is that we are counting near_deaths instead of df_deaths
    # which have a higher number .
    args[:mortality_rate] = args[:death_undercount_factor] * args[:county_mortality_rate] / args[:undercount_factor]
    (args[:mortality_rate] < 1.0) || error("mortality_rate is greater than one. check undercount factors.")

    # Rescale data to match graph size
    _, county_deaths_delt = get_data(args, county_deaths_cum * (args[:death_undercount_factor] / args[:county_pop] * num_nodes))
    _, county_infected_delt = get_data(args, county_infected_cum * (args[:undercount_factor] / args[:county_pop] * num_nodes))

    window = 7
    smooth_delt_deaths = rollmean(county_deaths_delt, window)
    smooth_delt_infected = rollmean(county_infected_delt, window)
    smooth_deaths = cumsum(smooth_delt_deaths)
    smooth_infected = cumsum(smooth_delt_infected)
    
    # We initialize infection on the graph with 5% uniform random people
    # We have already scaled counts by now. Therefore, begin at 5% of num_nodes
    # NOTE - we may use more specific initial infection later
    # N = 38 # findfirst(map(x-> x >= 0.05*num_nodes, smooth_infected))
    println(args[:io], ("smooth infected", smooth_infected))
    println(args[:io], ("inf_thresh_after_lead", args[:inf_thresh_after_lead]))
    println(args[:io], ("county_pop", args[:county_pop]))


    #   inf >= Th * pop
    #   
    #   inf >= Th * pop <==> 
    #   inf * UC / pop * NN >=  Th * pop * UC / pop * NN <==>
    #   inf * UC / pop * NN >=  Th  * UC  * NN
    #   smooth_inf >=  Th  * UC  * NN
    #   
    #   However, we can use 
    #        smooth_inf >=  Th  * NN
    #   since, having inflated the infection numbers, we'll reach a reasonable scale earlier.
    args[:start_day] = findfirst(map(x -> x >= args[:inf_thresh_after_lead]  * num_nodes, smooth_infected)) - args[:lead_in_time]
    args[:end_day] = args[:start_day] + args[:duration]
    ((args[:start_day] !== nothing) && (args[:start_day] > 0)) || error("Start day not found.  Using higher threshhold or smaller lead time")
    
    final_inf = smooth_infected[args[:start_day]:args[:end_day]]
    final_deaths = smooth_deaths[args[:start_day]:args[:end_day]]

    println(args[:io], ("First day of trajectories:", args[:start_day]))
    println(args[:io], ("smooth_deaths:", final_deaths))
    println(args[:io], ("smooth_infected", final_inf))
    
    (maximum(final_inf) < num_nodes) || error("More than 100 % Infection. max inf = $(maximum(final_inf)), num nodes = $(num_nodes)  Lower undercount_factor.")
    (maximum(final_deaths) < num_nodes) || error("More than 100 % Infection. max inf = $(maximum(final_deaths)), num nodes = $(num_nodes)  Lower undercount_factor.")
    return cat(final_inf, final_deaths, dims=[2])
end
