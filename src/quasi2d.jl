using Statistics
# using Peaks

export quasi2d_num_branches 
export quasi2d_smoothed_intensity
export quasi2d_histogram_intensity
export quasi2d_get_stats
export sample_midpoints 

"""
    sample_midpoints(a,b, n)

Divide the interval [a,b] in n equally sized segments and return a Vector of
midpoints of each segment.
"""
function sample_midpoints(a::Real, b::Real, n::Integer)::LinRange{Float64,Int64}
    @assert n > 0
    dx::Float64 = (b - a) / n
    first = a + dx / 2
    last = b - dx / 2
    return LinRange(first, last, n)
end


function quasi2d_num_branches(num_rays, dt, ts, potential::AbstractPotential;
    return_rays=false, rays_span = (0,1))
    ray_y = Vector(LinRange(rays_span[1], rays_span[2], num_rays + 1)[1:num_rays])
    return quasi2d_num_branches(ray_y, dt, ts, potential; return_rays=return_rays)
end

function quasi2d_num_branches(ray_y::AbstractVector{<:Real}, dt::Real,
    ts::AbstractVector{<:Real}, potential::AbstractPotential; return_rays=false)

    # HACK: This uses a lot of memory. Run GC here to try
    # limit memory use before big allocations.
    GC.gc(false)

    ray_y = Vector{Float64}(ray_y)
    num_rays = length(ray_y)
    ray_py = zeros(num_rays)
    num_branches = zeros(length(ts))
    ti = 1
    x = 0.0
    while ti <= length(ts)
        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # drift
        ray_y .+= dt .* ray_py
        x += dt
        if ts[ti] <= x
            num_branches[ti] = count_branches(ray_y)
            ti += 1
        end
    end
    if return_rays
        return num_branches, ray_y, ray_py
    else
        return num_branches
    end
end

"""
    quasi2d_smoothed_intensity(num_rays, dt, xs, ys, potential; b=0.0030, periodic_bnd = false)

Perform quasi-2D ray tracing simulation and return a matrix of smoothed
intensities.
Returned matrix has size (length(ys), length(xs)).

The situation being modeled is an infinite vertical wavefront starting
at x=0 with p_y=0 and p_x=1 (constant). To cover the area defined by xs and ys,
the actual wavefront being simulated must be taller than ys.
"""
function quasi2d_smoothed_intensity(num_rays::Integer, dt, xs, ys, potential; b=0.0030, periodic_bnd = false)
    h = length(ys) * (ys[2] - ys[1])
    τ = 0 
    if periodic_bnd == true
        rmin = ys[1]; rmax = ys[end]; τ = ys[end] - ys[1]
    else
        rmin,rmax = quasi2d_compute_front_length(1024, dt, xs, ys, potential)
    end
    ray_y = LinRange(rmin, rmax, num_rays)
    sim_h = (ray_y[2]-ray_y[1]) * length(ray_y)
    ints = quasi2d_smoothed_intensity(ray_y, dt, xs, ys, potential, b; periodic_bnd, τ) * (sim_h / h)
    return ints, (rmin, rmax)
end

function quasi2d_smoothed_intensity(
    ray_y::AbstractVector{<:Real},
    dt::Real,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    potential,
    b; 
    periodic_bnd = false, 
    τ = 0 
)
    dy = ys[2] - ys[1]
    rmin = ys[1]; rmax = ys[end]
    y_end = ys[1] + length(ys) * (ys[2] - ys[1])
    ray_y = Vector{Float64}(ray_y)
    num_rays = length(ray_y)
    ray_py = zeros(num_rays)
    xi = 1; x = xs[1]
    intensity = zeros(length(ys), length(xs))
    boundary = (
        ys[1] - dy - 4*b,
        ys[end] + dy + 4*b,
    )
    while xi <= length(xs)
        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # drift
        ray_y .+= dt .* ray_py
        x += dt
        if periodic_bnd == true
            for k in eachindex(ray_y)
                while ray_y[k] < rmin; ray_y[k] += τ; end
                while ray_y[k] > rmax; ray_y[k] -= τ; end
            end
        end
        while xi <= length(xs) && xs[xi] <= x
            # Compute intensity
            density = kde(ray_y, bandwidth=b, npoints=16 * 1024, boundary=boundary)
            intensity[:, xi] = pdf(density, ys)
            xi += 1
        end
    end
    return intensity
end

# First shoot some rays to figure out how wide they spread in the given time.
# Then use this to decide how tall to make the initial wavefront.
function quasi2d_compute_front_length(num_canary_rays, dt, xs::AbstractVector{<:Real}, ys, potential)
    T = xs[end] - xs[1]
    return quasi2d_compute_front_length(num_canary_rays, dt, T, ys, potential)
end

function quasi2d_compute_front_length(num_canary_rays, dt, T::Real, ys, potential)
    canary_ray_y = Vector(sample_midpoints(ys[1], ys[end], num_canary_rays))
    ray_y = copy(canary_ray_y)
    ray_py = zero(ray_y)
    for x ∈ 0:dt:T
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        ray_y .+= dt .* ray_py
    end
    max_travel = maximum(abs.(ray_y - canary_ray_y))
    # Front coordinates:
    rmin = ys[1] - max_travel
    rmax = ys[end] + max_travel
    return rmin, rmax
end
            
     

"""
    quasi2d_histogram_intensity(ray_y, xs, ys, potential)

Runs a simulation across a grid defined by `xs` and `ys`, returning a
matrix of flow intensity computed as a histogram (unsmoothed).
"""
function quasi2d_histogram_intensity(num_rays, xs, ys, potential; normalized = true, periodic_bnd = false)
    dt = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    
    # Compute the spread beforehand
    if periodic_bnd == true
        rmin = ys[1]; rmax = ys[end]; T = ys[end] - ys[1]
    else
        rmin,rmax = quasi2d_compute_front_length(1024, dt, xs, ys, potential)
    end

    width = length(xs)
    height = length(ys)
    image = zeros(height, width)
    ray_y = collect(LinRange(rmin, rmax, num_rays))
    ray_py = zeros(num_rays)
    for (xi, x) ∈ enumerate(xs)
        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # drift
        ray_y .+= dt .* ray_py
        # Collect
        for y ∈ ray_y
            if periodic_bnd == true
                while y < rmin; y += T; end
                while y > rmax; y -= T; end
            end
            yi = 1 + round(Int, (y - ys[1]) / dy)
            if yi >= 1 && yi <= height
                image[yi, xi] += 1
            end
        end
    end
    if normalized 
        for k in 1:length(xs)
            image[:,k] .= image[:,k]/sum(image[:,k])/dy
        end
    end
    # return image * height / num_rays
    return image 
end


# This function computes average area and the maximum intensity
function quasi2d_get_stats(num_rays::Integer, dt, xs::AbstractVector, ys::AbstractVector, potential; b=0.003, threshold = 1.5, periodic_bnd = false)
    τ = 0. 
    if periodic_bnd == true
        rmin = ys[1]; rmax = ys[end]; τ = ys[end] - ys[1]
    else
        rmin,rmax = quasi2d_compute_front_length(1024, dt, xs[end], ys, potential)
    end
    ray_y = LinRange(rmin, rmax, num_rays)
    area, max_I, nb_pks = quasi2d_smoothed_intensity_stats(ray_y, dt, xs, ys, potential, b, threshold, periodic_bnd, τ) 
    return area, max_I, nb_pks, (rmin, rmax)
end

# In this version you provide dt, T and x0. 
function quasi2d_get_stats(num_rays::Integer, dt, T::Real, ys::AbstractVector, potential; b=0.003, threshold = 1.5, x0 = 0, periodic_bnd = false)
    τ = 0. 
    if periodic_bnd == true
        rmin = ys[1]; rmax = ys[end]; τ = ys[end] - ys[1]
    else
        rmin,rmax = quasi2d_compute_front_length(2*1024, dt, T, ys, potential)
    end
    ray_y = LinRange(rmin, rmax, num_rays)
    xs = range(x0, T+x0, step = dt) 
    area, max_I, nb_pks= quasi2d_smoothed_intensity_stats(ray_y, dt, xs, ys, potential, b, threshold, periodic_bnd, τ) 
    return xs, area, max_I, nb_pks, (rmin, rmax)
end

function quasi2d_smoothed_intensity_stats(
    ray_y::AbstractVector{<:Real},
    dt::Real,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    potential::AbstractPotential,
    b::Real, 
    threshold::Real, 
    periodic_bnd::Bool,
    τ::Real 
)
    dy = step(ys)
    rmin = ys[1]; rmax = ys[end];
    h = length(ys) * dy
    num_rays = length(ray_y)
    sim_h = (ray_y[2]-ray_y[1]) * num_rays
    ray_y = Vector{Float64}(ray_y)
    ray_py = zeros(num_rays)
    area = zeros(length(xs))
    nb_pks = zeros(length(xs))
    max_I = zeros(length(xs))
    intensity = zeros(length(ys))
    background = (num_rays/length(ys)); 
    bckgnd_density = background/(dy*num_rays)
    xi = 1; x = xs[1]
    boundary = (
        ys[1] - dy - 4*b,
        ys[end] + dy + 4*b,
    )
    while x <= xs[end] + dt


        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # drift
        ray_y .+= dt .* ray_py
        x += dt

        if periodic_bnd == true
            for k in eachindex(ray_y)
                while ray_y[k] < rmin; ray_y[k] += τ; end
                while ray_y[k] > rmax; ray_y[k] -= τ; end
            end
        end

        while xi <= length(xs) &&  xs[xi] <= x 
            dy_ray = diff(ray_y)/dy 
            ind = findall(abs.(dy_ray) .< 2.)
            density = kde(ray_y[ind], bandwidth=b, npoints=16 * 1024, boundary=boundary)
            intensity = pdf(density, ys)*(sim_h / h)
            
            t = findmaxima(intensity)
            ind = findall(t[2] .> threshold) 
            nb_pks[xi] = length(ind)

            ind = findall(intensity .> threshold) 
            area[xi] = length(ind)/length(intensity)  
            max_I[xi] = maximum(intensity)  
            xi += 1
        end
    end
    return area, max_I, nb_pks
end

