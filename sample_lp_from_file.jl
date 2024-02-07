"""
    sample_from_file(filepath::String,num_samples::Int)

Draw a (small) sample of lamination parameters with according stacking sequences.

The function draws a random sample from a file, that is generated with `generate_file_for_sampling`
in `generate_random_lp.jl`, according to the weights saved in the file.

Due to the application of the kernel density estimator, a small sample is expected to have
a quasi-uniform distribution within the lamination parameter space.

# Arguments
- `filepath::String`: Complete filepath of the file to sample from
- `num_samples::Int`: number of points

# Returns
- `Vector{Int}`: Indices of the points in the file `filepath`, as a vector of length `num_samples`
- `Matrix{Float64}`: According lamination parameters, as an array of shape `(8,num_samples)`
- `Matrix{Int}`: According stacking sequences, as an array of shape `(num_plies,num_samples)`
"""
function sample_from_file(filepath::String,num_samples::Int)
    # load data
    fid = h5open(filepath,"r")
    lp_list = read(fid["sample/lamination_parameters"])
    stack_list = read(fid["sample/stacking_sequence"])
    weights_list = read(fid["sample/weights"])
    close(fid)

    # generate sample
    idx_list = sample(1:length(weights_list), Weights(weights_list), num_samples; replace=false)
    return idx_list, lp_list[:,idx_list], stack_list[:,idx_list]
end