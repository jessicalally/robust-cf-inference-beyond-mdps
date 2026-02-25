abstract type TerminationCriteria end
function termination_criteria(spec::Specification)
    prop = system_property(spec)
    return termination_criteria(prop, Val(isfinitetime(prop)))
end

struct FixedIterationsCriteria{T <: Integer} <: TerminationCriteria
    n::T
end
(f::FixedIterationsCriteria)(V, k, u) = k >= f.n
termination_criteria(prop, finitetime::Val{true}) =
    FixedIterationsCriteria(time_horizon(prop))

struct CovergenceCriteria{T <: AbstractFloat} <: TerminationCriteria
    tol::T
end
(f::CovergenceCriteria)(V, k, u) = maximum(abs, u) < f.tol
termination_criteria(prop, finitetime::Val{false}) =
    CovergenceCriteria(convergence_eps(prop))

"""
    solve(problem::AbstractIntervalMDPAlgorithm, alg::RobustValueIteration; callback=nothing)

Solve minimizes/maximizes optimistic/pessimistic specification problems using value iteration for interval Markov processes. 

It is possible to provide a callback function that will be called at each iteration with the current value function and
iteration count. The callback function should have the signature `callback(V::AbstractArray, k::Int)`.

`solve` can be called without specifying the algorithm, in which case it defaults to [`RobustValueIteration`](@ref).

### Examples

```jldoctest
prob1 = IntervalProbabilities(;
    lower = [
        0.0 0.5
        0.1 0.3
        0.2 0.1
    ],
    upper = [
        0.5 0.7
        0.6 0.5
        0.7 0.3
    ],
)

prob2 = IntervalProbabilities(;
    lower = [
        0.1 0.2
        0.2 0.3
        0.3 0.4
    ],
    upper = [
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ],
)

prob3 = IntervalProbabilities(;
    lower = [0.0; 0.0; 1.0],
    upper = [0.0; 0.0; 1.0]
)

transition_probs = [prob1, prob2, prob3]
initial_state = 1
mdp = IntervalMarkovDecisionProcess(transition_probs, initial_state)

terminal_states = [3]
time_horizon = 10
prop = FiniteTimeReachability(terminal_states, time_horizon)
spec = Specification(prop, Pessimistic, Maximize)

### Verification
problem = VerificationProblem(mdp, spec)
sol = solve(problem, RobustValueIteration(); callback = (V, k) -> println("Iteration ", k))
V, k, res = sol  # or `value_function(sol), num_iterations(sol), residual(sol)`

# Control synthesis
problem = ControlSynthesisProblem(mdp, spec)
sol = solve(problem, RobustValueIteration(); callback = (V, k) -> println("Iteration ", k))
σ, V, k, res = sol # or `strategy(sol), value_function(sol), num_iterations(sol), residual(sol)`
```
"""
function solve(problem::VerificationProblem, alg::RobustValueIteration; kwargs...)
    V, k, res, _ = _value_iteration!(problem, alg; kwargs...)
    return VerificationSolution(V, res, k)
end

function solve(problem::ControlSynthesisProblem, alg::RobustValueIteration; kwargs...)
    V, k, res, strategy_cache = _value_iteration!(problem, alg; kwargs...)
    strategy = cachetostrategy(strategy_cache)

    return ControlSynthesisSolution(strategy, V, res, k)
end

# TODO: currently only handles reward-based problems.
function solve_max_changes(problem::ControlSynthesisProblem, max_changes::Int, observed_actions; kwargs...)
    V, k, res, strategy_cache = _value_iteration_max_changes!(problem, max_changes, observed_actions; kwargs...)
    solutions = Vector{Any}(undef, max_changes + 2)

    mp = system(problem)
    spec = specification(problem)
    prop = system_property(spec)
    time_horizon = prop.time_horizon

    # V= [num states, T, max changes]
    
    for max_change in 0:max_changes
        solutions[max_change+1] = (strategy_cache[:, :, max_change+1], V[:, time_horizon, max_change+1], res[:, time_horizon, max_change+1], k)
    end
    
    return solutions
end

function _value_iteration!(problem::AbstractIntervalMDPProblem, alg; callback = nothing)
    mp = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(spec)
    upper_bound = isoptimistic(spec)
    maximize = ismaximize(spec)

    # It is more efficient to use allocate first and reuse across iterations
    workspace = construct_workspace(mp)
    strategy_cache = construct_strategy_cache(problem)

    value_function = ValueFunction(problem)
    initialize!(value_function, spec)
    nextiteration!(value_function)

    step!(
        workspace,
        strategy_cache,
        value_function,
        0,
        mp;
        upper_bound = upper_bound,
        maximize = maximize,
    )

    step_postprocess_value_function!(value_function, spec)
    step_postprocess_strategy_cache!(strategy_cache)
    k = 1

    if !isnothing(callback)
        callback(value_function.current, k)
    end

    while !term_criteria(value_function.current, k, lastdiff!(value_function))
        nextiteration!(value_function)

        step!(
            workspace,
            strategy_cache,
            value_function,
            k,
            mp;
            upper_bound = upper_bound,
            maximize = maximize,
        )
        step_postprocess_value_function!(value_function, spec)
        step_postprocess_strategy_cache!(strategy_cache)
        k += 1

        if !isnothing(callback)
            callback(value_function.current, k)
        end
    end

    postprocess_value_function!(value_function, spec)

    # lastdiff! uses previous to store the latest difference
    # and it is already computed from the condition in the loop
    return value_function.current, k, value_function.previous, strategy_cache
end

function _value_iteration_max_changes!(
    problem::AbstractIntervalMDPProblem,
    max_changes::Int,
    observed_actions;
    callback = nothing
)
    # V= [num states, T, max changes]
    max_changes = max_changes+1
    T = max_changes
    mp = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(spec)
    upper_bound = isoptimistic(spec)
    maximize = ismaximize(spec)

    # Allocate workspace
    workspace = construct_workspace(mp)

    # Allocate strategy caches.
    mp = system(problem)
    num_states = Int(product_num_states(mp)[1])

    strategy_cache = zeros(Float64, num_states, T+1, max_changes+1)

    # [num_states, T, max_changes+1]
    value_function = ValueFunctionMaxChanges(problem, T+1, max_changes+1)
    
    # Repeats reward correctly for new shape.
    initialize_max_actions!(value_function, spec)

    final_k = 0

    # Copies from previous to current.
    nextiteration!(value_function)

    for j in 0:max_changes
        step_max_changes!(
            workspace,
            strategy_cache,
            value_function,
            0,
            mp;
            upper_bound = upper_bound,
            maximize = maximize,
            max_changes = j,
            observed_actions = observed_actions,
            prop = system_property(spec)
        )

        step_postprocess_value_function_max_changes!(value_function, 0, j, spec)
    end
    
    k = 1

    if !isnothing(callback)
        callback(value_function.current, k)
    end


    while !term_criteria(value_function.current, k, lastdiff!(value_function))
        nextiteration!(value_function)
        
        for j in 0:max_changes
            step_max_changes!(
                workspace,
                strategy_cache,
                value_function,
                k,
                mp;
                upper_bound = upper_bound,
                maximize = maximize,
                max_changes = j,
                observed_actions = observed_actions,
                prop = system_property(spec)
            )

            step_postprocess_value_function_max_changes!(value_function, k, j, spec)
        end

        k += 1

        if !isnothing(callback)
            callback(value_function.current, k)
        end

    end

    return value_function.current, final_k, value_function.previous, strategy_cache
end


struct ValueFunction{R, A <: AbstractArray{R}}
    previous::A
    current::A
end

function ValueFunction(problem::AbstractIntervalMDPProblem)
    mp = system(problem)
    previous = arrayfactory(mp, valuetype(mp), product_num_states(mp))
    current = copy(previous)

    return ValueFunction(previous, current)
end

struct ValueFunctionMaxChanges{R}
    previous::Array{R,3}        # [num_states, T, max_changes+1] for DP updates
    current::Array{R,3}         # [num_states, T, max_changes+1]
end

function ValueFunctionMaxChanges(problem::AbstractIntervalMDPProblem, T::Int, max_changes::Int)
    mp = system(problem)
    num_states = Int(product_num_states(mp)[1])
    
    previous = zeros(Float64, num_states, T, max_changes)
    current  = zeros(Float64, num_states, T, max_changes)

    return ValueFunctionMaxChanges(previous, current)
end


function lastdiff!(V)
    # Reuse prev to store the latest difference
    V.previous .-= V.current
    rmul!(V.previous, -1.0)

    return V.previous
end

function nextiteration!(V)
    copy!(V.previous, V.current)

    return V
end

function step!(
    workspace,
    strategy_cache::OptimizingStrategyCache,
    value_function,
    k,
    mp;
    upper_bound,
    maximize,
)
    bellman!(
        workspace,
        strategy_cache,
        value_function.current,
        value_function.previous,
        mp;
        upper_bound = upper_bound,
        maximize = maximize,
    )
end

function step!(
    workspace,
    strategy_cache::NonOptimizingStrategyCache,
    value_function,
    k,
    mp;
    upper_bound,
    maximize,
)
    bellman!(
        workspace,
        strategy_cache[time_length(strategy_cache) - k],
        value_function.current,
        value_function.previous,
        mp;
        upper_bound = upper_bound,
        maximize = maximize,
    )
end

function step_max_changes!(
    workspace,
    strategy_cache,
    value_function,
    k,
    mp;
    upper_bound,
    maximize,
    max_changes,
    observed_actions,
    prop
)
    bellman_max_changes!(
        workspace,
        strategy_cache,
        value_function.current,
        value_function.previous,
        k,
        mp;
        upper_bound = upper_bound,
        maximize = maximize,
        max_changes = max_changes,
        observed_actions,
        prop
    )
end
