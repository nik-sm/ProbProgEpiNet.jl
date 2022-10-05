function expose_by_comm(g::SEIRGraph, v::Int64, E0_pcts::Array{Float64})
    p = E0_pcts[get_prop(g,v,:community)]
    return([1-p, p, 0, 0])
end

function make_komplett_graph(n_nodes)
    n_community = 1
    g = SEIRGraph()
    defaultweight!(g, 0)
        
    # average edge weight from los-angeles 01:
    # w = 0.03659008016823566
    w = 1.0

    for i in 1:n_nodes
        add_vertex!(g, Dict(
                    :id        => "$(i)",
                    :community => 1,
                    :household => 1,
                    :status    => S))
        
        for j in 1:i-1
            add_edge!(g, i, j, Dict(
                    :type   => "household",
                    :weight => w))
        end
    end
    
    set_prop!(g, :num_communities,          n_community)
    set_prop!(g, :weight_params,            nothing)
    set_prop!(g, :house_weight,             nothing)
    set_prop!(g, :vertex_community_indices, nothing)
    set_prop!(g, :community_sizes,          nothing)
    set_prop!(g, :initial_adjacency,        nothing)
    set_prop!(g, :community_graph,          nothing)
    
    return g
end
