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


include(srcdir("utils.jl"))

function compute_lyap(d)
    @unpack T, dt, V, y_init = d # parametros
    λ = _get_lyap_1D(d) 
    return @strdict(λ)
end

function get_branch_number(v0, V, y_init, xs, num_rays, j, K, threshold, T; prefix = "fermi_dec") 
    dt = step(xs)
    d = @dict(V, v0, y_init, dt, T, j, num_rays)  
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        compute_lyap, # function
        prefix = "lyap_fermi", # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )

    @unpack λ = data
    d = @dict(V,v0, y_init, T, threshold, num_rays, xs, λ, j, K)  
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        compute_branches, # function
        prefix = prefix, # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    # @unpack nb_br = data
    return data
end

function compute_branches(d)
    @unpack V, y_init, xs, K, λ, threshold = d
    ind = findall(λ .> threshold) 
    y_i = y_init[ind]
    dy = step(y_init)
    nb_br_pos, hst_pos = manifold_track(xs, ys, y_i, dy, V; K)

    ind = findall(λ .<threshold) 
    y_i = y_init[ind]
    nb_br_z, hst_z = manifold_track(xs, ys, y_i, dy, V; K)

    nb_br_all, hst_all = manifold_track(xs, ys, y_init, dy, V; K)
    return @strdict(nb_br_z, hst_z, nb_br_pos, hst_pos, nb_br_all, hst_all)
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
    # height = length(ys)
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



v0 = 0.1; dt = 0.01; num_rays = 200000; n_avg = 40; K = 1.;
threshold = 1e-3; T = 1500
y_init = range(-40*0.2, 40*0.2, length = num_rays)
ys = y_init[findall(0 .≤ y_init .≤ 1.)]
xs = range(0,15., step = dt)



a = 0.2; dot_radius = 0.2*0.25; softness = 0.2; θ_range = range(0,π/4, length = n_avg) 
hst_v_all = Vector{Vector{Float64}}()
hst_v_pos = Vector{Vector{Float64}}()
hst_v_z = Vector{Vector{Float64}}()
nb_v_all = Vector{Vector{Float64}}()
nb_v_pos = Vector{Vector{Float64}}()
nb_v_z = Vector{Vector{Float64}}()
for j = 1:n_avg
    V = LatticePotential(a*rotation_matrix(θ_range[j]), dot_radius, v0; softness=softness)
    dat = get_branch_number(v0, V, y_init, xs, num_rays, j, K, threshold, T; prefix = "fermi_br_lyap") 
    @unpack hst_all, hst_pos, hst_z = dat
    @unpack nb_br_all, nb_br_pos, nb_br_z = dat

    push!(hst_v_all, hst_all)
    push!(hst_v_z, hst_z)
    push!(hst_v_pos, hst_pos)
    push!(nb_v_all, nb_br_all)
    push!(nb_v_z, nb_br_z)
    push!(nb_v_pos, nb_br_pos)
end

pargs = (yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 30) 
fig = Figure(size=(800, 1200))
ax1= Axis(fig[2, 1]; xlabel = L"t",  ylabel = L"f_{area}",  pargs...) 
ax2= Axis(fig[1, 1]; xlabel = L"t", ylabel = L"N_{b}", pargs...) 
lines!(ax1, xs, mean(hst_v_pos, dims = 1)[1]; label =L"\lambda > 0", color = :green)
lines!(ax1, xs, mean(hst_v_z, dims = 1)[1]; color = :red, label = L"\lambda \simeq 0")
lines!(ax1, xs, mean(hst_v_all, dims = 1)[1]; color = :blue, label = "all rays")
ylims!(ax1,0.,0.2)
xlims!(ax1,0.,15)
axislegend(ax1; labelsize = 30)

lines!(ax2, xs, mean(nb_v_pos, dims = 1)[1], label =L"\lambda > 0", color=:green)
lines!(ax2, xs, mean(nb_v_z, dims = 1)[1]; color = :red,label = L"\lambda \simeq 0")
lines!(ax2, xs, mean(nb_v_all, dims = 1)[1]; color = :blue, label = "all rays")
xlims!(ax2,0.,15)
axislegend(ax2; labelsize = 30)

Label(fig[1, 1, TopLeft()], "(a)", padding = (0,15,15,0), fontsize = 30)
Label(fig[2, 1, TopLeft()], "(b)", padding = (0,15,15,0), fontsize = 30)
s = "fermi_lyap_branch_num.pdf"
save(plotsdir(s),fig)
