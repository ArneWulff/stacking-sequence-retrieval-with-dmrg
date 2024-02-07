"""
File for generating the set of target lamination parameters.

It shows, how we used the relevant functions to generate a large sample of 
target lamination parameters, and estimate the point density with a kernel density
estimation. This can then be used to generate smaller samples, that are quasi-uniformly
distributed in lamination parameter space.
"""

using HDF5
using LinearAlgebra

include("laminationparameters.jl")
include("generate_random_stacks.jl")
include("generate_random_lp.jl")


let 
    # file directory and identifier
    filelocation = "..."
    filename = "sample_lp_disor_constr"

    # settings

    angles = [0, 45, 90, -45]

    num_angles = length(angles)
    max_num_segments = 10
    num_plies = 200
    num_samples = 10000

    distance = 45 + 1
    c_list = disorientation_constraint_violations(angles,distance)

    batch_size = 10000

    h = 0.005

    batches = Vector(0:49)

    # generate stacking sequences
    for n ∈ batches
        generate_lp_sample_batch(
            filelocation, filename,
            angles, num_plies,
            distance,
            c_list
        ) 
    end

    # calculate sums for KDE
    for n ∈ batches
        calculate_kernel_sums_batch(
            filelocation, filename, n,batches, h
        )
    end

    # calculate weights for points
    calculate_weights(
        filelocation,filename,
        batches
    )

    # summerize points and weights in one file
    generate_file_for_sampling(
        filelocation, filename, batches
    )

end
