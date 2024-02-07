"""
Generate a set of target lamination parameters

Example of usage: `run_generate_random_lp.jl`

Any random generator for stacking sequences will generally create
a non-uniform point distribution in lamination parameter space.
Here we use a kerndel density estimator to get closer to a uniform
distribution.

First we generate a large number of random stacking sequences (in 
our case 500000), which produces an equivalent number of points in
lamination parameter space. We then use a kernel density estimator
to approximate the point density at each generated point.
The inverse of this point density is then used as weights for each
point, in order to sample a smaller number of points (here 100)
in a way that is closer to a uniform distribution.

Since the kernel density estimation becomes quite expensive, we
generated smaller batches of points (here 10000), that we can
process individually.

Function `generate_lp_sample_batch` generates one batch of 
stacking sequences.
File structure for the batches of generated stacking sequences
with corresponding lamination parameters:
🗂️ HDF5.File:
├─ 📂 properties
│  ├─ 🏷️ batch_num : batch index
│  ├─ 🏷️ batch_size : batch size
│  ├─ 🏷️ disorientation_constraint_distance
|  |        angle distance for disorientation constraint
│  ├─ 🏷️ num_angles : number of possible ply angles
│  ├─ 🏷️ num_plies : number of plies
│  ├─ 🔢 angles : possible ply angles
│  └─ 🔢 disorientation_constraint_vecs
|           table for constraint violations of ply angles
|
└─ 📂 sample
   ├─ 🔢 lamination_parameters
   |  |      lamination parameters of the generated 
   |  |      stacking sequences
   │  ├─ 🏷️ cols = "lp"
   │  └─ 🏷️ rows = "sample_number"
   └─ 🔢 stacking_sequence : generated stacking sequences
      ├─ 🏷️ cols = "stack"
      └─ 🏷️ rows = "sample_number"

Function `calculate_kernel_sums_batch` calculates the sums within
the kernel density estimator between the different batches. The
function creates a new file.
Function `calculate_weights` calculates the weights from these sums.
The function adds the weights to the existing file
File structure of the generated file:
🗂️ HDF5.File:
├─ 📂 data
│  ├─ 🔢 batch_is_calculated : 1 if the batch is processed, 0 else
│  │  └─ 🏷️ rows = "sample_number"
│  ├─ 🔢 kernels
|  |        resulting kernels: sum of the kernel sums, divided by the
|  |        number of points
│  ├─ 🔢 kernels_batches
|  |        which batches are included in the kernels. 1 if the
|  |        batch is included, 0 else
│  ├─ 🔢 kernelsums
|  |  |     contributions of each batch to the point density at each
|  |  |     point, according to the KDE
│  │  ├─ 🏷️ cols = "sample_number"
│  │  ├─ 🏷️ h : variance used in the kernel
│  │  └─ 🏷️ rows = "kernel_sums"
│  └─ 🔢 weights : resulting weights for sampling (inverse point density)
|
└─ 📂 properties : are the same as before
   ├─ 🏷️ batch_num : batch index
   ├─ 🏷️ batch_size : batch size
   ├─ 🏷️ disorientation_constraint_distance
   |        angle distance for disorientation constraint
   ├─ 🏷️ num_angles : number of possible ply angles
   ├─ 🏷️ num_plies : number of plies
   ├─ 🔢 angles : possible ply angles
   └─ 🔢 disorientation_constraint_vecs
            table for constraint violations of ply angles

Function `generate_file_for_sampling` summerizes the relevant information,
the lamination parameters, stacks and weights, in a single file, from which
smaller samples can be easily generated according to the weights.
File structure of the generated file:
🗂️ HDF5.File:
├─ 📂 properties
│  ├─ 🏷️ batch_num = 0 (unused, from copying properties from previous files)
│  ├─ 🏷️ batch_size : size of a single batch (irrelevant)
│  ├─ 🏷️ disorientation_constraint_distance : angle distance of disorientation constraint
│  ├─ 🏷️ kernel_h : variance in KDE
│  ├─ 🏷️ num_angles : number of possible ply angles
│  ├─ 🏷️ num_plies : number of plies
│  ├─ 🔢 angles : possible ply angles
│  └─ 🔢 disorientation_constraint_vecs : table for constriant violations for angle pairs
|
└─ 📂 sample : all lamination parameters with stacking sequences and weights for sampling
   ├─ 🔢 lamination_parameters
   ├─ 🔢 stacking_sequence
   └─ 🔢 weights


"""

#%%

"""
    create_datafile_sample_lp_batch(
    filepath::String, batch_num::Int,
    angles::Vector{<:Real}, num_plies::Integer,
    disorientation_constraint_distance::Real,
    disorientation_constraint_vecs::BitMatrix
)

Create file for a batch of possible target lamination parameters.

# Arguments
- `filepath::String`: complete filepath (including filename) for the file
- `batch_num::Int`: batch index
- `angles::Vector{<:Real}`: possible ply_angles
- `num_plies::Integer`: number of plies
- `disorientation_constraint_distance::Real`: angle distance in disorientation constraint
- `disorientation_constraint_vecs::BitMatrix`: table with constraint violations
"""
function create_datafile_sample_lp_batch(
    filepath::String, batch_num::Int,
    angles::Vector{<:Real}, num_plies::Integer,
    disorientation_constraint_distance::Real,
    disorientation_constraint_vecs::BitMatrix
)
    @assert !isfile(filepath) "File already exists."
    fid = h5open(filepath, "cw")
    props = create_group(fid,"properties")
    attributes(props)["batch_num"] = batch_num
    props["angles"] = angles
    attributes(props)["num_angles"] = num_angles
    attributes(props)["num_plies"] = num_plies
    attributes(props)["disorientation_constraint_distance"] = disorientation_constraint_distance
    props["disorientation_constraint_vecs"] = convert.(Int8,disorientation_constraint_vecs)
    close(fid)
end


"""
    generate_lp_sample_batch(
    filelocation::String, filename::String,
    angles::Vector{<:Real}, num_plies::Integer,
    disorientation_constraint_distance::Real,
    disorientation_constraint_vecs::BitMatrix;
    batch_size::Int=10000,max_num_segments::Int=10
)

Generate one batch of stacking sequences with according lamination parameters.

The file is generated under the name "filelocation/filename_batch_XXXX.h5py", with XXXX
being the batch number. The batch number is automatically determined by considering
the already present files in the directory.

For a more diverse range of stacking sequences, the number of segments for generating
the stacking sequences (`generate_stack_from_probs`) is chosen randomly from
1:max_num_segments. Further, it is chosen randomly, from which end the angles are
placed (`reverse=true` or `false`).

# Arguments
- `filelocation::String`: File directory
- `filename::String`: File identifier
- `angles::Vector{<:Real}`: possible ply angles
- `num_plies::Integer`: number of plies
- `disorientation_constraint_distance::Real`: angle distance for disorientation constraint
- `disorientation_constraint_vecs::BitMatrix`: constraint violations as a table

# Keywords
- `batch_size::Int=10000`: number of generated points per batch
- `max_num_segments::Int=10`: maximum number of segments for generating the stacking sequence
"""
function generate_lp_sample_batch(
    filelocation::String, filename::String,
    angles::Vector{<:Real}, num_plies::Integer,
    disorientation_constraint_distance::Real,
    disorientation_constraint_vecs::BitMatrix;
    batch_size::Int=10000,max_num_segments::Int=10
)   
 
    if filelocation[end] ≠ '\\'
        filelocation *= "\\"
    end

    # get already computed batches in folder
    files_in_folder = readdir(filelocation)
    files_in_folder = files_in_folder[(x -> startswith(x,filename*"_batch") && endswith(x,".hdf5")).(files_in_folder)]

    # get last highest batch number of folder, plus 1
    batch_num = length(files_in_folder) > 0 ? (maximum((x -> parse(Int,split(splitext(x)[1],'_')[end])).(files_in_folder)) + 1) : 0

    # create file
    filepath = filelocation*filename*"_batch_"*lpad(batch_num,4,"0")*".hdf5"
    create_datafile_sample_lp_batch(
        filepath, batch_num, angles, num_plies,
        disorientation_constraint_distance,
        disorientation_constraint_vecs
    )

    # create arrays to store generated LP and stacking sequences
    lp_list = Matrix{Float64}(undef,8,batch_size)
    stack_list = Matrix{Int}(undef,num_plies,batch_size)

    t1 = time()
    for s ∈ 1:batch_size
        # generate stacking sequence
        probs = generate_probs_stack(num_angles, rand(1:max_num_segments))
        reverse = rand([false,true])
        stack = generate_stack_from_probs(
            num_plies,probs,disorientation_constraint_vecs;reverse=reverse
        )

        # calculate lamination parameters
        lp = lamination_parameters(angles[stack])

        # save results
        lp_list[:,s] = lp[:]
        stack_list[:,s] = stack[:]
    end
    t2 = time()

    # save results into previously created file
    fid = h5open(filepath, "r+")
    attributes(fid["properties"])["batch_size"] = batch_size
    sample = create_group(fid,"sample")
    sample["lamination_parameters"] = lp_list
    attributes(sample["lamination_parameters"])["rows"] = "sample_number"
    attributes(sample["lamination_parameters"])["cols"] = "lp"
    sample["stacking_sequence"] = stack_list
    attributes(sample["stacking_sequence"])["rows"] = "sample_number"
    attributes(sample["stacking_sequence"])["cols"] = "stack"
    close(fid)

    println("Batch $batch_num complete. Time: $(t2-t1)")

end

gaussian_kernel(x::Float64,h::Float64) = exp(-x^2/(2*h)) / √(2*h)

function gaussian_kernel(x::Vector{Float64},h::Float64)
    d = length(x)
    return exp(-(x⋅x)/(2*h)) / √(2*d*h)
end

"""
    kernel_sum(x::Vector{Float64},points::Matrix{Float64},h::Float64)

Calculate the sum, that is relevant for the kernel density estimation.

# Arguments
- `x::Vector{Float64}`: Point at which the point density is calculated
- `points::Matrix{Float64}`: Points which contribute to the point density
- `h::Float64`: Variance in the gaussian kernel

# Returns
- `Float64`: resulting sum (division by the number of points gives the
                point density)
"""
function kernel_sum(x::Vector{Float64},points::Matrix{Float64},h::Float64)
    # points: coords, samples
    # x .- points: add column vector x to every column vector in points
    # dims=1: apply over columns (index rows)
    return sum(mapslices(xi -> gaussian_kernel(xi,h),x .- points,dims=1))
end

"""
    create_datafile_lp_kernelsums_batch(filelocation::String,filename::String,batch_num::Int,h::Float64)

Create datafile for calculating the point densities and the weights.

The file is generated under the name "filelocation/filename_kernelsums_batch_XXXX.h5py",
with XXXX being the batch number. `filelocation` and `filename` must be consistent with
the files that store the generated lamination parameters and stacking sequences.

# Arguments
- `filelocation::String`: file directory
- `filename::String`: file identifier
- `batch_num::Int`: batch number
- `h::Float64`: variance in the kernel density estimation
"""
function create_datafile_lp_kernelsums_batch(filelocation::String,filename::String,batch_num::Int,h::Float64)
    if filelocation[end] ≠ '\\'
        filelocation *= "\\"
    end

    # filepath for stored lamination parameters and stacking sequences
    filepath = filelocation*filename*"_batch_"*lpad(batch_num,4,"0")*".hdf5"

    # check if file exists
    @assert isfile(filepath) "File $filename for batch $batch_num not found."

    # filepath for new file
    filepath_new = filelocation*filename*"_kernelsums_batch_"*lpad(batch_num,4,"0")*".hdf5"

    fid = h5open(filepath, "r")
    batch_size = read(attributes(fid["properties"])["batch_size"])
    fid_new = h5open(filepath_new,"cw")

    # copy properties
    props = create_group(fid_new,"properties")
    for ds ∈ keys(fid["properties"])
        props[ds] = read(fid["properties"][ds])
    end
    for attr ∈ keys(attributes(fid["properties"]))
        attributes(props)[attr] = read(attributes(fid["properties"])[attr])
    end
    close(fid)

    # create datasets
    data = create_group(fid_new,"data")
    ds_kernelsums = create_dataset(
        data,"kernelsums",Float64,((batch_num+1,batch_size),(-1,batch_size)), # (dims, max_dims)
        chunk=(100,batch_size) # chunk_dims 
    )
    ds_kernelsums[:,:] .= 0.
    attributes(data["kernelsums"])["cols"] = "sample_number"
    attributes(data["kernelsums"])["rows"] = "kernel_sums"
    attributes(data["kernelsums"])["h"] = h
    ds_bic = create_dataset(
        data,"batch_is_calculated",Int8,((batch_num+1,),(-1,)), # (dims, max_dims)
        chunk=(100,) # chunk_dims 
    )
    ds_bic[:] .= 0
    attributes(data["batch_is_calculated"])["rows"] = "sample_number"
    close(fid_new)
end


"""
    calculate_kernel_sums_batch(
    filelocation::String,filename::String,batch_num::Int,calculate_for_batches::Vector{Int},h::Float64;
    overwrite::Bool=true
)

Create file and calculate the partial sums for the kernel density estimation.

The arguments `filelocation` and `filename` must be constistent with the function `generate_lp_sample_batch`.

# Arguments
- `filelocation::String`: file directory
- `filename::String`: file identifier
- `batch_num::Int`: batch index for which the partial sums for the point densities are calculated
- `calculate_for_batches::Vector{Int}`: points that are included in the sum, and thus constribute to the point density
- `h::Float64`: variance in the kernel density estiamtion

# Keywords
- `overwrite::Bool=true`: whether to replace already existing values with new calculations
"""
function calculate_kernel_sums_batch(
    filelocation::String,filename::String,batch_num::Int,calculate_for_batches::Vector{Int},h::Float64;
    overwrite::Bool=true
)
    if filelocation[end] ≠ '\\'
        filelocation *= "\\"
    end

    filepath_data = filelocation*filename*"_batch_"*lpad(batch_num,4,"0")*".hdf5"
    filepath = filelocation*filename*"_kernelsums_batch_"*lpad(batch_num,4,"0")*".hdf5"

    # get alrady computed batches in folder
    files_in_folder = readdir(filelocation)
    files_in_folder = files_in_folder[(x -> startswith(x,filename*"_batch") && endswith(x,".hdf5")).(files_in_folder)]

    # get all available batch numbers of samples
    available_batches = (x -> parse(Int,split(splitext(x)[1],'_')[end])).(files_in_folder)

    # check if sample for batch exists
    @assert batch_num ∈ available_batches "Batch $batch_num not found."

    # check if all batches for calculation exist
    @assert calculate_for_batches ⊆ available_batches "Some batches in $calculate_for_batches not found."

    # check if file exists. If not, create file
    if !isfile(filepath)
        create_datafile_lp_kernelsums_batch(filelocation,filename,batch_num,h)
    end
    @assert isfile(filepath)

    # If overwrite false, get uncalculated batches
    if overwrite
        fid = h5open(filepath,"r")
        batch_is_calculated = BitVector(read(fid["data"]["batch_is_calculated"]))
        close(fid)
        setdiff!(calculate_for_batches,(0:(length(batch_is_calculated)-1))[batch_is_calculated])
    end

    # extend dataset if needed
    fid = h5open(filepath,"r+")
    @assert isapprox(read(attributes(fid["data/kernelsums"])["h"]),h,atol=1e-10) "Kernel size h of file ($h) and argument do not match."
    ds_num_batches = length(read(fid["data/batch_is_calculated"]))
    max_batch_num = maximum(calculate_for_batches) + 1
    if ds_num_batches < max_batch_num
        HDF5.set_extent_dims(fid["data/batch_is_calculated"],(max_batch_num,))
        HDF5.set_extent_dims(fid["data/kernelsums"],(max_batch_num,batch_size))
    end
    close(fid)

    # load data from current batch
    fid = h5open(filepath_data,"r")
    batch_lps = read(fid["sample/lamination_parameters"])
    close(fid)

    # allocate memory
    data_tmp = zeros(Float64,batch_size)

    # calculate kernel sums
    t1 = time()
    for (bi,b) ∈ enumerate(calculate_for_batches)
        # load data from batch
        fp = filelocation*filename*"_batch_"*lpad(b,4,"0")*".hdf5"
        fid = h5open(fp,"r")
        b_lps = read(fid["sample/lamination_parameters"])
        close(fid)
        
        # calculate kernel_sums
        data_tmp = mapslices(x -> kernel_sum(x,b_lps,h), batch_lps, dims=1)[:]
        println("Batch $b calculated. Time: $(time()-t1)")
        
        # save data
        fid = h5open(filepath,"r+")
        fid["data/kernelsums"][b+1,:] = data_tmp # batch number 0-indexed
        fid["data/batch_is_calculated"][b+1] = Int8[1]
        close(fid)
        println("Batch $b saved. Time: $(time()-t1)")
    end
end


#%%

function calculate_weights(
    filelocation::String,filename::String,
    batches::Vector{Int}
)
    calculate_weights(filelocation,filename,batches,batches)
end


"""
    calculate_weights(
    filelocation::String,filename::String,
    batches::Vector{Int},calculate_for_batches::Vector{Int}
)

Calculate the point densities and weights from the partial sums.
The results are added to the existing files from the `kernelsums`.

The arguments `filelocation` and `filename` must be consistent with
the functions `generate_lp_sample_batch` and `calculate_kernel_sums_batch`.

# Arguments
- `filelocation::String`: file directory
- `filename::String`: file identifier
- `batches::Vector{Int}`: batches for points for which the weights are calculated
- `calculate_for_batches::Vector{Int}`: batches that constribute to the point density
"""
function calculate_weights(
    filelocation::String,filename::String,
    batches::Vector{Int},calculate_for_batches::Vector{Int}
)
    if filelocation[end] ≠ '\\'
        filelocation *= "\\"
    end

    # get alrady computed batches in folder
    files_in_folder = readdir(filelocation)
    files_in_folder = files_in_folder[(x -> startswith(x,filename*"_kernelsums_batch") && endswith(x,".hdf5")).(files_in_folder)]

    # get all available batch numbers of samples
    available_batches = (x -> parse(Int,split(splitext(x)[1],'_')[end])).(files_in_folder)

    # check if batches exist
    @assert batches ⊆ available_batches "Some batches not found."
    @assert calculate_for_batches ⊆ available_batches "Some 'batches in calculate_for_batches' not found."

    for batch ∈ batches
        # load data for batch
        # get calculated batches
        filepath = filelocation*filename*"_kernelsums_batch_"*lpad(batch,4,"0")*".hdf5"
        fid = h5open(filepath,"r")
        ksums = read(fid["data/kernelsums"])
        b_is_calc = Bool.(read(fid["data/batch_is_calculated"]))[:]
        close(fid)

        cfb = zero(b_is_calc)
        for b ∈ calculate_for_batches
            cfb[b+1] = b_is_calc[b+1]
        end
        ksums = ksums[cfb,:]
        num_batches,num_samples = size(ksums)

        # for each calculate sum, divide by batchsize*num_batches
        kernels = sum(ksums,dims=1)/(num_batches*num_samples)
        weights = 1 ./ kernels

        # save
        fid = h5open(filepath,"r+")
        fid["data/kernels_batches"] = Int8.(cfb)
        fid["data/kernels"] = kernels
        fid["data/weights"] = weights
        close(fid)
    end

end


"""
    generate_file_for_sampling(
    filelocation::String, filename::String, batches::Vector{Int}
)

Create one file with the lamination parameters, stacking sequences and 
weights for sampling.

The arguments `filelocation` and `filename` must be consistent with 
the functions before. The new file is created as
`filelocation/filename__full_weighted_sample.hdf5`.

# Arguments
- `filelocation::String`: file directory
- `filename::String`: file identifier
- `batches::Vector{Int}`: batches to include
"""
function generate_file_for_sampling(
    filelocation::String, filename::String, batches::Vector{Int}
)
    if filelocation[end] ≠ '\\'
        filelocation *= "\\"
    end

    # get alrady computed batches in folder
    files_in_folder = readdir(filelocation)
    # only need to check "kernelsums" files, since they necesitate the original samples
    files_in_folder = files_in_folder[(x -> startswith(x,filename*"_kernelsums_batch") && endswith(x,".hdf5")).(files_in_folder)]

    # get all available batch numbers of samples
    available_batches = (x -> parse(Int,split(splitext(x)[1],'_')[end])).(files_in_folder)

    # check if batches exist
    @assert batches ⊆ available_batches "Some batches not found."

    # get batch size
    filepath = filelocation*filename*"_batch_"*lpad(batches[1],4,"0")*".hdf5"
    fid = h5open(filepath,"r")
    batch_size = read(attributes(fid["properties"])["batch_size"])
    num_plies = read(attributes(fid["properties"])["num_plies"])
    close(fid)

    num_batches = length(batches)
    sample_size = num_batches*batch_size

    # space for lps, stacks and weights
    lp_list = zeros(Float64,8,sample_size)
    stack_list = zeros(Int,num_plies,sample_size)
    weights_list = zeros(Float64,sample_size)
    
    for bidx ∈ 1:num_batches
        batch = batches[bidx]
        idx1 = (bidx-1)*batch_size + 1
        idx2 = bidx*batch_size

        # load lps and stacks
        filepath = filelocation*filename*"_batch_"*lpad(batch,4,"0")*".hdf5"
        fid = h5open(filepath,"r")
        lp_list[:,idx1:idx2] = read(fid["sample/lamination_parameters"])
        stack_list[:,idx1:idx2] = read(fid["sample/stacking_sequence"])
        close(fid)

        # load weights
        filepath = filelocation*filename*"_kernelsums_batch_"*lpad(batch,4,"0")*".hdf5"
        fid = h5open(filepath,"r")
        weights_list[idx1:idx2] = read(fid["data/weights"])
        close(fid)
    end

    # normalize weights
    weights_list = weights_list / sum(weights_list)
    
    # create file 
    filepath_new = filelocation*filename*"_full_weighted_sample.hdf5"
    fid_new = h5open(filepath_new,"cw")

    # copy properties
    filepath = filelocation*filename*"_kernelsums_batch_"*lpad(batches[1],4,"0")*".hdf5"
    fid = h5open(filepath,"r")
    props = create_group(fid_new,"properties")
    for ds ∈ keys(fid["properties"])
        props[ds] = read(fid["properties"][ds])
    end
    for attr ∈ keys(attributes(fid["properties"]))
        attributes(props)[attr] = read(attributes(fid["properties"])[attr])
    end
    attributes(props)["kernel_h"] = read(attributes(fid["data/kernelsums"])["h"])
    close(fid)

    # save data
    data = create_group(fid_new,"sample")
    data["lamination_parameters"] = lp_list
    data["stacking_sequence"] = stack_list
    data["weights"] = weights_list

    close(fid_new)
end