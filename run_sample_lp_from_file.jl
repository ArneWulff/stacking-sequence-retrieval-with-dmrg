"""
File for creating a set of target lamination parameters, that are
quasi-uniformly distributed in lamination parameter space.
"""

using HDF5
using StatsBase: sample, Weights
using Plots
using LinearAlgebra: norm

include("sample_lp_from_file.jl")

let 
    # file location and name of file with lp and weights
    filelocation = "..."
    filename = "sample_lp_disor_constr_full_weighted_sample.hdf5"
    filepath = filelocation*filename

    # number of samples to be drawn
    num_samples = 100

    # sample from file
    idx_list, lp_list, stack_list = sample_from_file(filepath,num_samples)

    # create new file for sample of target parameters
    fid_new = h5open(filelocation*"lp_sample_100.hdf5","cw")

    # copy properties
    fid = h5open(filepath,"r")
    props = create_group(fid_new,"properties")
    for ds ∈ keys(fid["properties"])
        props[ds] = read(fid["properties"][ds])
    end
    for attr ∈ keys(attributes(fid["properties"]))
        attributes(props)[attr] = read(attributes(fid["properties"])[attr])
    end
    close(fid)

    # save lp,stacks
    data = create_group(fid_new,"sample")
    data["lamination_parameters"] = lp_list
    data["stacking_sequence"] = stack_list
    data["idx_list"] = idx_list

    close(fid_new)
end