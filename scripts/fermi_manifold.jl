using DrWatson 
@quickactivate
include(srcdir("BranchedFlowSim.jl"))
using .BranchedFlowSim
using ProgressMeter
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StatsBase
using CodecZlib
using KernelDensity
using ProgressMeter

include(srcdir("utils.jl"))


function get_branch_number(v0, y_init, xs, num_rays, K, T; prefix = "fermi_dec", force = false, a, dot_radius, softness, θ_range) 
    d = @dict(v0, y_init, num_rays, xs, K, a, dot_radius, T, softness, θ_range)  
    
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        compute_branches; # function
        prefix, # prefix for savename
        force, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    return data
end

function compute_branches(d)
    @unpack  y_init, xs, K, v0,  num_rays, a, dot_radius, softness, θ_range = d

    hst_v_all = Vector{Vector{Float64}}(undef, n_avg)
    nb_v_all = Vector{Vector{Float64}}(undef, n_avg)
    @showprogress Threads.@threads for j = 1:n_avg
        V = LatticePotential(a*rotation_matrix(θ_range[j]), dot_radius, v0; softness)
        dy = step(y_init)
        nb_v_all[j], hst_v_all[j] = manifold_track(xs, ys, y_init, dy, V; K)
        # push!(hst_v_all, hst_all)
        # push!(nb_v_all, nb_br_all)
    end
    return @strdict(nb_v_all, hst_v_all)
end

function count_branches(y_ray, dy, ys; K = 1.)
    dy_ray = diff(y_ray)/dy
    hst = zeros(length(ys))
    a = ys[1]; b = ys[end]; dys = ys[2] - ys[1]
    ind = findall(abs.(dy_ray) .< K)
    str = find_consecutive_streams(ind) 
    br = 0; 
    for s in str 
        if length(s) ≥ 3
            # if all(x -> (a ≤ x ≤ b) , y_ray[s])
                br += 1
            # end
            for y in y_ray[s]
                yi = 1 + round(Int, (y - a) / dys)
                if yi >= 1 && yi <= length(ys)
                    hst[yi] = 1
                end
            end
        end
    end
    return br, sum(hst)/length(hst)
end


function manifold_track(xs, ys, y_init, dyr, potential; K = 1.)
    dt = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    num_rays = length(y_init)
    width = length(xs)
    hst = zeros(width)
    nb_br = zeros(width)
    ray_y = deepcopy(collect(y_init))
    ray_py = zeros(num_rays)
    xi = 1; x = xs[1]
    while x <= xs[end] + dt
        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # drift
        ray_y .+= dt .* ray_py
        x += dt
        while xi <= length(xs) &&  xs[xi] <= x 
            nb_br[xi], hst[xi] = count_branches(ray_y, dyr, ys)
            xi += 1
        end
    end
    return  nb_br, hst 
end



v0 = 0.1; dt = 0.01; num_rays = 200000; n_avg = 40; K = 1.; T = 100
y_init = range(-40*0.2, 40*0.2, length = num_rays)
ys = y_init[findall(0 .≤ y_init .≤ 1.)]
xs = range(0,T, step = dt)
v_range = logrange(0.001, 1, length = 20)
a = 0.2; dot_radius = 0.2*0.25; softness = 0.2; θ_range = range(0,π/4, length = n_avg) 

for v0 in v_range
    dat = get_branch_number(v0, y_init, xs, num_rays, K, T; prefix = "fermi_br_lyap", 
                           a, dot_radius, softness, θ_range, force = false) 
    @unpack hst_v_all,nb_v_all = dat
    pargs = (yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 30) 
    fig = Figure(size=(800, 1200))
    ax2= Axis(fig[1, 1]; xlabel = L"t", ylabel = L"N_{b}", pargs...) 
    lines!(ax2, xs, mean(nb_v_all, dims = 1)[1]; color = :blue, label = "all rays")
    save(plotsdir(savename("plot_nb",@dict(v0, num_rays), "pdf")),fig)
    println("\\includegraphics[width=0.3\\textwidth]{../plots/",savename("plot_nb",@dict(v0, num_rays), "pdf"),"}")
end


