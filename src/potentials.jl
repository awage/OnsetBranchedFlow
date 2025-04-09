#=
File containing various 2D potential functions.
Functions are implemented as distinct types (structs) which
implement both the calling operator and force(x,y) function
returning the force for a mass=1 particle at a given coordinate.
=#
using LinearAlgebra
using StaticArrays
using FFTW
using Interpolations
# using KernelDensity
using LaTeXStrings
using Random:Xoshiro

export AbstractPotential
export FermiDotPotential, LatticePotential, PeriodicGridPotential
export CompositePotential, RepeatedPotential
export RotatedPotential, TranslatedPotential
export CosSeriesPotential
export FunctionPotential
export ZeroPotential
export CosMixedPotential
export StdMapPotential

export rotation_matrix, force, force_x, force_y
export correlated_random_potential
export random_fermi_potential
export fermi_dot_lattice_cos_series
export complex_separable_potential
export shaken_fermi_lattice_potential
export grid_eval

"""
We consider an object V to be a Potential if it is callable with two floats
(like V(x,y)) and has a defined function force(V, x, y) = -∇V. Plain functions
are supported with numerical differentiation, and some of the potentials
defined here can be composed together (LatticePotential, CompositePotential)
to create complex yet efficient potential functions.
"""
function ispotential(v)
    return applicable(v, 1.2, 2.3) && applicable(force, v, 1.2, 2.3)
end

# NOTE: AbstractPotential is used as a subtype for potentials with custom force
# functions,
abstract type AbstractPotential end;
# This is the most abstract potential 

struct FermiDotPotential <: AbstractPotential
    radius::Float64
    α::Float64
    inv_α::Float64
    v0::Float64
    function FermiDotPotential(radius, v0, softness=0.2)
        α = softness * radius
        return new(radius, α, 1 / α, v0)
    end
end

function (V::FermiDotPotential)(x::Real, y::Real)::Float64
    d = V.radius - sqrt(x^2 + y^2)
    return V.v0 / (1 + exp(-V.inv_α * d))
end

function force(V::FermiDotPotential, x::Real, y::Real)::SVector{2,Float64}
    d = sqrt(x^2 + y^2)
    if d <= 1e-9
        return SVector(0.0, 0.0)
    end
    power = V.inv_α * (-V.radius + d)
    z = @inline exp(power)
    return SVector(x, y) * (V.v0 * z * V.inv_α / (d * (1 + z)^2))
end

"""
LatticePotential

"""
struct LatticePotential{DotPotential<:AbstractPotential} <: AbstractPotential
    A::SMatrix{2,2,Float64,4}
    A_inv::SMatrix{2,2,Float64,4}
    dot_potential::DotPotential
    offset::SVector{2,Float64}
    four_offsets::SVector{4,SVector{2,Float64}}
    function LatticePotential(A, radius, v0; offset=[0, 0], softness=0.2)
        A = SMatrix{2,2}(A)
        dot = FermiDotPotential(radius, v0, softness)
        fo = [SVector(0.0, 0.0), A[:, 1], A[:, 2], A[:, 1] + A[:, 2]]
        return new{FermiDotPotential}(
            A,
            SMatrix{2,2}(inv(A)),
            dot, SVector{2,Float64}(offset), fo)
    end
    function LatticePotential(A, dot::AbstractPotential; offset=[0,0])
        A = SMatrix{2,2}(A)
        fo = [SVector(0.0, 0.0), A[:, 1], A[:, 2], A[:, 1] + A[:, 2]]
        return new{typeof(dot)}(
            A,
            SMatrix{2,2}(inv(A)),
            dot, SVector{2,Float64}(offset), fo)
    end
end

"""
    rotation_matrix(θ)

Returns a 2D rotation matrix for the given angle θ.
"""
function rotation_matrix(θ)
    return [
        cos(θ) -sin(θ)
        sin(θ) cos(θ)
    ]
end

function (V::LatticePotential)(x::Real, y::Real)::Float64
    # Find 4 closest lattice vectors
    r = SVector(x, y) - V.offset
    a::SVector{2,Float64} = V.A_inv * r
    # ind = SVector{2, Float64}(floor(a[1]), floor(a[2]))
    ind::SVector{2,Float64} = floor.(a)
    v = 0.0
    R0 = V.A * ind
    for offset ∈ V.four_offsets
        R = R0 + offset
        v += V.dot_potential(r[1] - R[1], r[2] - R[2])
    end
    return v
end

function force(V::LatticePotential, x::Real, y::Real)::SVector{2,Float64}
    # Find 4 closest lattice vectors
    r = SVector{2,Float64}(x - V.offset[1], y - V.offset[2])
    a = (V.A_inv * r)::SVector{2,Float64}
    # ind = SVector{2, Float64}(floor(a[1]), floor(a[2]))
    ind::SVector{2,Float64} = floor.(a)
    F::SVector{2,Float64} = SVector(0.0, 0.0)
    R0::SVector{2,Float64} = V.A * ind
    for offset ∈ V.four_offsets
        rR::SVector{2,Float64} = r - (R0 + offset)
        F += @inline force(V.dot_potential, rR[1], rR[2])::SVector{2,Float64}
    end
    return F
end

function force_y(V::LatticePotential, x::Float64, y::Float64)
    return force(V, x, y)[2]
end

struct PeriodicGridPotential{Interpolation<:AbstractInterpolation} <: AbstractPotential
    itp::Interpolation
    """
    PeriodicGridPotential(xs, ys, arr)

Note: `arr` is indexed as arr[y,x].
"""
    function PeriodicGridPotential(xs, ys, arr::AbstractMatrix{Float64})
        itp = interpolate(transpose(arr), BSpline(Cubic(Periodic(OnCell()))))
        itp = extrapolate(itp, Periodic(OnCell()))
        itp = scale(itp, xs, ys)
        return new{typeof(itp)}(itp)
    end
end

function (V::PeriodicGridPotential)(x::Real, y::Real)
    return V.itp(x, y)
end

"""
    force(V::PeriodicGridPotential, x::Real, y::Real)

"""
function force(V::PeriodicGridPotential, x::Real, y::Real)
    return -gradient(V.itp, x, y)
end

function force_x(V, x, y)
    return force(V, x, y)[1]
end
function force_y(V, x, y)
    return force(V, x, y)[2]
end

"""
    force_diff(V, x::Real, y::Real)

Generic implementation of `force` function using numeric differentiation.
Should only be used for testing.
"""
function force_diff(V, x::Real, y::Real)::SVector{2,Float64}
    h = 1e-6
    return -SVector(
        (@inline V(x + h, y) - @inline V(x - h, y)) / (2 * h),
        (@inline V(x, y + h) - @inline V(x, y - h)) / (2 * h)
    )
end

"""
    force(V::Function, x::Real, y::Real)

Generic implementation of `force` function for potentials defined as plain
functions. Uses numerical differentiation
"""
function force(V::Function, x::Real, y::Real)::SVector{2,Float64}
    return force_diff(V, x, y)
end

function force_y(V::Function, x::Real, y::Real)::Float64
    h = 1e-6
    return -(@inline V(x, y + h) - @inline V(x, y - h)) / (2 * h)
end

function force_x(V::Function, x::Real, y::Real)::Float64
    h = 1e-6
    return -(@inline V(x + h, y) - @inline V(x - h, y)) / (2 * h)
end

"""
    compare_force_with_diff(p)

Debugging function for comparing `force` implementation with `force_diff`
"""
function compare_force_with_diff(p)
    xs = LinRange(0, 1, 99)
    ys = LinRange(0, 1, 124)
    return [
        norm(force_diff(p, x, y) - force(p, x, y)) for y ∈ ys, x ∈ xs
    ]
    # fig, ax, plt = heatmap(xs, ys, (x, y) -> norm(force_diff(p, x, y) - force(p, x, y)))
    # Colorbar(fig[1, 2], plt)
    # fig
end

function count_zero_crossing(fx)
    c = 0
    for j ∈ 2:length(fx)
        if sign(fx[j-1]) != sign(fx[j])
            c += 1
        end
    end
    return c
end

function count_branches(ray_y)
    caustics = 0
    for j ∈ 2:length(ray_y)-1
        if sign(ray_y[j] - ray_y[j-1]) != sign(ray_y[j+1] - ray_y[j])
            caustics += 1
        end
    end
    return caustics / 2
end

struct RotatedPotential{OrigPotential<:AbstractPotential} <: AbstractPotential
    A::SMatrix{2,2,Float64,4}
    A_inv::SMatrix{2,2,Float64,4}
    V::OrigPotential
    function RotatedPotential(θ::Real, V::AbstractPotential)
        rot = rotation_matrix(θ)
        return new{typeof(V)}(rot, inv(rot), V)
    end
end

function (V::RotatedPotential)(x::Real, y::Real)::Float64
    x, y = V.A * SVector(x, y)
    return V.V(x, y)
end

function force(V::RotatedPotential, x::Real, y::Real)::SVector{2,Float64}
    x, y = V.A * SVector(x, y)
    F = force(V.V, x, y)
    return V.A_inv * F
end


struct CompositePotential{DotPotential} <: AbstractPotential
    dot_potentials::Vector{DotPotential}
    potential_size::Float64
    locations::Matrix{Float64}
    grid_x::Float64
    grid_y::Float64
    # grid_locations::Matrix{Vector{SVector{2,Float64}}}
    grid_indices::Matrix{Vector{Int32}}
    grid_w::Int64
    grid_h::Int64

    function CompositePotential(locations::AbstractMatrix,
        dot_potential::AbstractPotential, potential_size::Real)
        potentials = Vector{typeof(dot_potential)}(
            [dot_potential for _ ∈ eachcol(locations)]
        )
        return CompositePotential(locations, potentials, potential_size)
    end

    function CompositePotential(locations::AbstractMatrix,
        dot_potentials::AbstractVector{DotPotential}, potential_size::Real) where {DotPotential <: AbstractPotential}
        min_x, max_x = extrema(locations[1, :])
        min_y, max_y = extrema(locations[2, :])
        # XXX: Explain this math
        grid_w = 3 + ceil(Int, (max_x - min_x) / potential_size)
        grid_h = 3 + ceil(Int, (max_y - min_y) / potential_size)
        grid_indices= [
            Vector{Int32}()
            for y ∈ 1:grid_h, x ∈ 1:grid_w
        ]
        offsets = [
            0 0
            0 1
            1 0
            0 -1
            -1 0
            -1 -1
            -1 1
            1 1
            1 -1
        ]'
        for (i, (x, y)) ∈ enumerate(eachcol(locations))
            ix = 2 + floor(Int, (x - min_x) / potential_size)
            iy = 2 + floor(Int, (y - min_y) / potential_size)
            for (dx, dy) ∈ eachcol(offsets)
                push!(grid_indices[iy+dy, ix+dx], i)
            end
        end
        return new{DotPotential}(dot_potentials, potential_size,
            locations, min_x, min_y, grid_indices,
            grid_w, grid_h
        )
    end
end

function (V::CompositePotential)(x::Real, y::Real)::Float64
    # find index
    ix = 2 + floor(Int, (x - V.grid_x) / V.potential_size)
    iy = 2 + floor(Int, (y - V.grid_y) / V.potential_size)
    if ix < 1 || iy < 1 || ix > V.grid_w || iy > V.grid_h
        return 0.0
    end
    v = 0
    @inbounds for idx ∈ V.grid_indices[iy, ix]
        @inbounds rx = V.locations[1, idx]
        @inbounds ry = V.locations[2, idx]
        @inbounds pot = V.dot_potentials[idx]
        v += @inline pot(x - rx, y - ry)
    end
    return v
end

function force(V::CompositePotential, x::Real, y::Real)::SVector{2,Float64}
    # find index
    ix = 2 + floor(Int, (x - V.grid_x) / V.potential_size)
    iy = 2 + floor(Int, (y - V.grid_y) / V.potential_size)
    if ix < 1 || iy < 1 || ix > V.grid_w || iy > V.grid_h
        return SVector(0.0, 0.0)
    end
    F = SVector(0.0, 0.0)
    for idx ∈ V.grid_indices[iy, ix]
        @inbounds rx = V.locations[1, idx]
        @inbounds ry = V.locations[2, idx]
        @inbounds pot = V.dot_potentials[idx]
        F += force(pot, x - rx, y - ry)
    end
    return F
end

# TODO: get rid of this old name
const RepeatedPotential = CompositePotential;

struct CosSeriesPotential{MatrixType} <: AbstractPotential
    w::MatrixType
    k::Float64
    function CosSeriesPotential(w::AbstractMatrix{Float64}, k)
        s = size(w)
        @assert s[1] == s[2]
        return new{SMatrix{s[1],s[1],Float64,s[1]^2}}(
            w, k
        )
    end
end

# function compute_sincos_kx(n, k, x)
# end

function (V::CosSeriesPotential)(x, y)
    len = size(V.w)[1]
    xcos = MVector{len,Float64}(undef)
    ycos = MVector{len,Float64}(undef)
    xcos[1] = ycos[1] = 1.0
    # TODO: Optimize this. Not as important as `force`, which is called much
    # more often.
    for k ∈ 2:len
        kk = (k - 1) * V.k
        xcos[k] = cos(kk * x)
        ycos[k] = cos(kk * y)
    end
    xycos = ycos * transpose(xcos)
    return sum(
        V.w .* xycos
    )
end

function force(V::CosSeriesPotential, x, y)
    # Uncomment for debug 
    # return BranchedFlowSim.force_diff(V, x, y)
    # Compute gradient
    len = size(V.w)[1]
    # Following arrays will have values like
    # xcos[i] = cos((i-1)*k*x) and so on.
    xcos = MVector{len,Float64}(undef)
    ycos = MVector{len,Float64}(undef)
    xsin = MVector{len,Float64}(undef)
    ysin = MVector{len,Float64}(undef)
    xcos[1] = ycos[1] = 1.0
    xsin[1] = ysin[1] = 0.0

    # Sin and cos only need to be computed once, after that we can use the
    # sum angle formula to compute the rest.
    sinkx, coskx = sincos(V.k * x)
    sinky, cosky = sincos(V.k * y)
    xcos[2] = coskx
    ycos[2] = cosky
    xsin[2] = sinkx
    ysin[2] = sinky

    for k ∈ 3:len
        # Angle sum formulas applied here
        xcos[k] = xcos[k-1] * coskx - xsin[k-1] * sinkx
        ycos[k] = ycos[k-1] * cosky - ysin[k-1] * sinky
        xsin[k] = xsin[k-1] * coskx + xcos[k-1] * sinkx
        ysin[k] = ysin[k-1] * cosky + ycos[k-1] * sinky
    end
    # Modify sin terms to compute the derivative
    for k ∈ 2:len
        kk = (k - 1) * V.k
        xsin[k] *= -kk
        ysin[k] *= -kk
    end
    F = SVector(0.0, 0.0)
    for i ∈ 1:len
        for j ∈ 1:len
            c = V.w[i, j]
            F += c * SVector(xsin[i] * ycos[j], ysin[i] * xcos[j])
        end
    end
    return -F
end

struct FunctionPotential{F} <: AbstractPotential
    f::F
end

function (V::FunctionPotential)(x::Real, y::Real)::Float64
    return V.f(x, y)
end

function force(V::FunctionPotential, x::Real, y::Real)::SVector{2,Float64}
    return force_diff(V.f, x, y)
end

function Base.convert(::Type{AbstractPotential}, fun::Function)
    if !applicable(fun, 1.2, 3.14)
        throw(DomainError(fun, "Is not a function taking 2 floats"))
    end
    return FunctionPotential{typeof(fun)}(fun)
end

struct TranslatedPotential{WrappedPotential <: AbstractPotential} <: AbstractPotential
    wrapped::WrappedPotential
    tx :: Float64
    ty :: Float64
end

function (V::TranslatedPotential)(x::Real, y::Real)::Float64
    return V.wrapped(V.tx + x, V.ty + y)
end

function force(V::TranslatedPotential, x::Real, y::Real)::SVector{2,Float64}
    return force(V.wrapped, V.tx + x, V.ty + y)
end

# Optimized potential for free particles
struct ZeroPotential <: AbstractPotential
end

function (V::ZeroPotential)(x::Real, y::Real)::Float64
    return 0.0
end

function force(V::ZeroPotential, x::Real, y::Real)::SVector{2,Float64}
    return SVector(0.0, 0.0)
end


## Various helper functions for creating potentials follow.
# Keep potential type definitions above this.


function fermi_dot_lattice_cos_series(degree, lattice_a, dot_radius, v0; softness=0.2)
    pot = LatticePotential(lattice_a * I, dot_radius, v0, softness=softness)
    # Find coefficients numerically by doing FFT
    N = 128
    xs = LinRange(0, lattice_a, N + 1)[1:N]
    g = grid_eval(xs, xs, pot)
    # Fourier coefficients
    w = (2 / length(g)) * fft(g)
    # Convert from exp coefficients to cosine coefficients
    w = real(w[1:(degree+1), 1:(degree+1)])
    w[1, 1] /= 2
    w[2:end, 2:end] *= 2
    for i ∈ 1:degree
        for j ∈ 1:degree
            if (i + j) > degree
                w[1+i, 1+j] = 0
            end
        end
    end
    k = 2pi / lattice_a
    # Use statically sized arrays.
    # static_w = SMatrix{1 + degree,1 + degree,Float64,(1 + degree)^2}(w)
    return CosSeriesPotential(w, k)
end

"""
    correlated_random_potential(width,
    height,
    correlation_scale,
    v0, seed=rand(UInt))

Returns a Gaussian correlated random potential (PeriodicGridPotential) with
periodicity (width,height), given `correlation_scale` and height `v0`. If
specified, `seed` is used for random numbers, so that repeated calls result
in the same potential (as long as other parameters don't change).

Returned potential satisfies the following
E[V(r)] = 0
E[V(r₁)*V(r₂)] = v₀² exp(-(|r₁-r₂|²/c²))
"""
function correlated_random_potential(width,
    height,
    correlation_scale,
    v0, seed=rand(UInt))
    # To generate the potential array, use a certain number dots per one correlation
    # scale.
    rand_N = 16 / correlation_scale
    rNy = round(Int, rand_N * height)
    rNx = round(Int, rand_N * width)
    ys = LinRange(0, height, rNy + 1)[1:end-1]
    xs = LinRange(0, width, rNx + 1)[1:end-1]
    pot_arr = v0 * gaussian_correlated_random(xs, ys, correlation_scale, seed)
    return PeriodicGridPotential(xs, ys, pot_arr)
end

"""
    gaussian_correlated_random(xs, ys, scale, seed=rand(UInt))

Returns random potential matrix of Gaussian correlated random values.
"""
function gaussian_correlated_random(xs, ys, scale, seed=rand(UInt))
    # See scripts/verify_correlated_random.jl for testing this
    rng = Xoshiro(seed)
    ymid = middle(ys)
    xmid = middle(xs)
    # TODO: Explain this. 
    # dist2 = ((ys .- ymid) .^ 2) .+ transpose()
    xcorr = exp.(-(xs .- xmid) .^ 2 ./ (scale^2))
    ycorr = exp.(-(ys .- ymid) .^ 2 ./ (scale^2))
    corr =  ComplexF64.(ycorr .* transpose(xcorr))
        # exp.( -dist2 ./ (scale^2)))
    # Convert DFT result to fourier series
    fcorr = fft!(corr)
    num_points = length(xs) * length(ys)
    # phase = rand(rng, length(ys), length(xs))
    # TODO: Not sure why the factor 2 is included here inside sqrt
    ft = num_points .* sqrt.((2/num_points).*fcorr) .* cis.(2pi .* rand.(rng))
    ifft!(ft)
    # vrand = ifft(num_points * sqrt.(fcorr) .* cis.(2pi .* phase))
    return real(ft)
end


"""
    random_fermi_potential(width, height, lattice_a, dot_radius, v0;
        softness=0.2)

Returns a potential with randomly placed Fermi bumps. Fermi bumps have
radius `dot_radius` and height `v0`. Average density of the bumps matches a
periodic lattice with lattice constant `lattice_a`.

Randomly generated potential has size (width,height) and is repeated with that
period to make a periodic potential.
"""
function random_fermi_potential(width, height, lattice_a, dot_radius, v0;
        softness=0.2)
    num_dots = round(Int, width * height / lattice_a^2)
    locs = zeros(2, num_dots)
    for i ∈ 1:num_dots
        locs[:, i] = [-width/2, -height/2] + rand(2) .* [width, height]
    end
    pot = CompositePotential(
        locs,
        FermiDotPotential(dot_radius, v0, softness),
        lattice_a
    )
    # Convert into a lattice potential to make it repeat forever
    A = [
        width 0
        0 height
    ]
    return LatticePotential(A, pot)
end

function complex_separable_potential(degree,
    lattice_a, dot_radius, v0; softness=0.2)
    zpot = fermi_dot_lattice_cos_series(degree, lattice_a, dot_radius, v0;
        softness=softness)
    w = Matrix(zpot.w)
    w[2:end, 2:end] .= 0
    return CosSeriesPotential(w, zpot.k)
end

"""
    shaken_fermi_lattice_potential(A, dot_radius, v0; pos_dev, v_dev,
    softness=0.2, period_n = 21)
    softness=0.2, period_n = 20)

Returns a perturbed Fermi lattice potential with each bump position randomly
displaced by a normal distribution with deviation `pos_dev`. Height of
each bump is perturbed by a normal distribution with deviation `v_dev`
(either deviation can be set to 0.)

Returned potential is made fully periodic by repeating the random 
"""
function shaken_fermi_lattice_potential(A, dot_radius, v0; pos_dev, v_dev,
    softness=0.2, period_n = 21)
    @assert period_n % 2 == 1

    potentials = FermiDotPotential[]
    locations = zeros(2, period_n^2)
    for i ∈ -period_n÷2 : period_n÷2
        for j ∈ -period_n÷2 : period_n÷2
            r = A * [i,j]
            r += pos_dev * randn(2)
            v = v0 + randn() * v_dev
            push!(potentials, FermiDotPotential(dot_radius, v, softness))
            idx = length(potentials)
            locations[:, idx] = r
        end
    end
    # With softness=0.2 the potential decays to 2e-9 at distance 5.
    dot_size = dot_radius + 3 * dot_radius * softness / 0.2 
    comp = CompositePotential(locations, potentials, dot_size)
    # Make it periodic
    return LatticePotential(
        period_n * A, comp
    )
end



"""
    grid_eval(xs, ys, fun)

Evaluates given callable `fun` in a grid defined by `xs` and `ys`
and returns a matrix of values. The returned matrix is oriented
such that y-axis is along rows and x-axis is along columns, i.e.
    grid_eval(xs, ys, fun)[i,j] = fun(xs[j], ys[i])
"""
function grid_eval(xs, ys, fun)
    return [
        fun(x, y) for y ∈ ys, x ∈ xs
    ]
end


struct CosMixedPotential <: AbstractPotential
    r::Float64
    a::Float64
    v0::Float64
end

function (V::CosMixedPotential)(x, y)
    return  V.v0*(-(1-V.r)/2*(cos(x*2π/V.a)+cos(y*2π/V.a))-V.r*cos(x*2π/V.a)*cos(y*2π/V.a))
end


function force(V::CosMixedPotential, x, y)
    xp = x*2π/V.a; yp = y*2π/V.a; 
    return SVector(0,-V.v0*2π/V.a*sin(yp)*(2*V.r*cos(xp)-V.r+1)/2)
    # return SVector(-V.v0*2π/V.a*sin(xp)*(2*V.r*cos(yp)-V.r+1)/2, 
    #                -V.v0*2π/V.a*sin(yp)*(2*V.r*cos(xp)-V.r+1)/2 )
end

struct StdMapPotential <: AbstractPotential
    a::Float64
    v0::Float64
end

function (V::StdMapPotential)(x, y)
    return  -V.v0*cos(y*2π/V.a)
end

function force(V::StdMapPotential, x, y)
    yp = y*2π/V.a;  
    return SVector(0, V.v0/(2π*V.a)*sin(yp))
end


