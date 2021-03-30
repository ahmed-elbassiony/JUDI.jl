# Parallel instance of lsrtm_objective function
# Author: Mathias Louboutin, mloubutin3@gatech.edu
# Date: September 202-
#

"""
    lsrtm_objective(model, source, dobs, dm; options=Options(), nlind=false, dprecon=nothing, mprecon=nothing)

Evaluate the least-square migration objective function. Returns a tuple with function value and \\
gradient. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and \\
observed data of type `judiVector`.

Example
=======

    function_value, gradient = lsrtm_objective(model, source, dobs, dm)

"""
function lsrtm_objective(model::Model, source::judiVector, dObs::judiVector, dm; options=Options(),
                         nlind=false, dprecon::Union{Nothing, joAbstractLinearOperator}=nothing,
                         mprecon::Union{Nothing, joAbstractLinearOperator}=nothing)
# lsrtm_objective function for multiple sources. The function distributes the sources and the input data amongst the available workers.
    !isnothing(mprecon) && (dm = mprecon*dm)
    results = judipmap(j -> lsrtm_objective(model, source[j], dObs[j], dm, subsample(options, j);
                                            nlind=nlind, dprecon=dprecon), 1:dObs.nsrc)

    # Collect and reduce gradients
    obj, gradient = reduce((x, y) -> x .+ y, results)
    isnothing(mprecon) && (gradient = adjoint(mprecon)*gradient)
    # first value corresponds to function value, the rest to the gradient
    return obj, gradient
end
