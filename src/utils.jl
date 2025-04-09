using ChaosTools
using LinearAlgebra:norm

function quasi2d_map!(du,u,p,t)
    y,py = u; potential, dt = p
    # kick
    du[2] = py + dt * force_y(potential, dt*t, y)
    # drift
    du[1] = y + dt * du[2]
    return nothing
end


function _get_lyap_1D(d) 
    @unpack V, dt, T, y_init = d
    df = DeterministicIteratedMap(quasi2d_map!, [0.4, 0.2], [V, dt])
    py = 0.
    res = length(y_init)
    λ1 = zeros(res)
    p = Progress(res, "Lyap calc") 
    Threads.@threads for k in eachindex(y_init)
        λ1[k] = lyapunov(df, T; u0 = [y_init[k], py]) 
        next!(p)
    end
    return λ1
end

function get_fit_lin(xg, yg)
    model(x, p) = p[1] .+ p[2] .* x
    p0 = [yg[1], 0.2]
    fit = curve_fit(model, xg, yg, p0)
    @show fit.param
    return fit.param, model, xg
end

function fit_lyap(xg, yg)
    model(x, p) = p[1] .+ p[2] *x
    p0 = [yg[1], 0.2]
    fit = curve_fit(model, xg, yg, p0)
    return fit.param, model
end

function get_fit_exp(xg, yg)
    model(x, p) = p[1] * exp.(p[2] * x)
        mx, indi = findmax(yg)
        xdata = xg[indi:end]
        ydata = yg[indi:end]
    lb = [0., -1.]
    ub = [2000, 0.]   
    p0 = [ydata[1], -0.2]
    fit = curve_fit(model, xdata, ydata, p0; lower = lb, upper = ub)
    @show fit.param
    return fit.param, model, xdata
end

function find_consecutive_streams(arr::Vector{Int})
    diffs = diff(arr)
    starts = [1; findall(diffs .!= 1) .+ 1]
    ends = [findall(diffs .!= 1); length(arr)]

    consecutive_streams = [arr[starts[i]:ends[i]] for i in 1:length(starts) if ends[i] - starts[i] + 1 > 1]

    return consecutive_streams
end
