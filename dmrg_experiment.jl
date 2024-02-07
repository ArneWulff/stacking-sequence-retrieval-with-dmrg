"""
Functions specific to the experiments performed for the paper

# Experiment description

We performed DMRG on the stacking sequence retrieval problem on several
target lamination parameters with a variety of settings, i. a. in the bond-dimension
and sweeping direction. Each target and setting gets its own file for the results.

For given settings and target lamination parameters, we performed DMRG a few times
with different initial MPS. These different trials are saved in the same file.

The DMRG function `dmrg_experiment_one_try` is a modified version of `dmrg_costum`
from `dmrg_costum.jl`, which enables costum sweeping direction sequences. The 
function `dmrg_experiment_one_try` has some additional modifications to record 
different kinds of data, like sampling of an intermediate state or recording the
duration of one sweep. This data is returned at the end of the function. The
function performs a single run of DMRG.

The function `dmrg_experiment` performs several trials of DMRG by calling the 
function `dmrg_experiment_one_try` several times. It also creates a file for
storing the results and data. This is the core function of the experiment.

For an example of usage, see `run_dmrg_experiment.jl`


# File format

The results are stored in an HDF5 file. File structure:
🗂️ HDF5.File:
├─ 📂 properties : Settings for the experiment
│  ├─ 🏷️ disorientation_constraint
|  |        1 or 0, whether the constraint is enforced
│  ├─ 🏷️ disorientation_constraint_distance
|  |        angle distance for constraint in degrees
│  ├─ 🏷️ disorientation_constraint_strength
|  |        strength of penalty for constraint
│  ├─ 🏷️ maxdim
|  |        maximum bond dimension for whole run
│  ├─ 🏷️ num_angles
|  |        number of possible angles
│  ├─ 🏷️ num_plies
|  |        number of plies in the stack
│  ├─ 🏷️ num_sweeps
|  |        number of sweeps in one run of the optimization
│  ├─ 🏷️ num_tries
|  |        number of trials with the same setting,
|  |        except for different initial MPS
│  ├─ 🏷️ sample_idx
|  |        sample index, corresponds to index of target
|  |        lamination parameters
│  ├─ 🏷️ sweep_sequence
|  |        used sweep sequence in the optimization
│  ├─ 🔢 angles
|  |        possible ply angles in degrees
│  ├─ 🔢 disorientation_constraint_matrix
|  |        BitMatrix for constraint violations of pairs of angles
│  ├─ 📂 sweeps : setting of the `Sweep` object
│  │  ├─ 🔢 cutoff
│  │  └─ 🔢 maxdim
│  ├─ 🔢 target_parameters
|  |        the target lamination parameters
│  └─ 🔢 target_stack
|           a stacking sequence producing the lamination parameters
|
├─ 📂 data : store data from experiments
│  ├─ 🔢 elapsed_time_sweeps
|  |        elapsed duration of each sweep
│  ├─ 🔢 energies
|  |        energy expectation value measured after each sweep
│  ├─ 🔢 maxlinkdim
|  |        setting of maximum bond dimension for each sweep
│  ├─ 🔢 time_stamps : time stamps after each sweep
|  └─ 📂 psi_samples
|     |     every couple of sweeps, the current MPS is sampled.
|     |     This provides a cross section of the superposition of the MPS
|     |     without the need to store the complete MPS, which can be quite large
│     ├─ 📂 try_1
│     │  ├─ 🔢 sweep_10
│     │  ├─ 🔢 sweep_20
│     │  ⋮
|     |
│     ├─ 📂 try_2
│     ├─ 📂 try_3
│     ⋮ 
|
├─ 📂 results
|  ├─ (🔢 constraint_violations
|  |        sum of constraint violations
|  |   ❗ due to an error, constraint violations were not counted and saved
|  |   ❗ in the HDF5 files)
|  ├─ 🔢 lamination_parameters
|  |        resulting lamination parameters
|  ├─ 🔢 loss
|  |        resulting loss (mean square error) to the target parameters
|  ├─ 🔢 rmse
|  |        resulting rmse to the target parameters
|  ├─ 🔢 stack
|  |        resulting stacking sequence
|  └─ 📂 psi
|           resulting MPS, compatible with ITensors way of
|           storing and loading an MPS
|     ├─ 📂 try_1
|     ├─ 📂 try_2
|     ⋮  
|
└─ 📂 psi0 : initial MPS, compatible with storing and loading
   |          an MPS in ITensors
   ├─ 📂 try_1
   ├─ 📂 try_2
   ⋮


      
      

"""

"""
    create_file_for_dmrg_experiment(
    filelocation::String, filename::String, sample_idx::Int,
    num_plies::Int,angles::Vector{<:Union{Int,Float64}},
    target_parameters::Vector{Float64},target_stack::Vector{Int},
    disorientation_constraint::Bool,c_strength::Float64,
    c_distance::Union{Float64,Int},c_list::Union{Matrix{<:Int},BitMatrix},
    psi0_list::Vector{MPS},psi0_energies_list::Vector{Float64},sweeps::Sweeps,sweep_sequence::String
)

Create an HDF5 file for the experiments

# Arguments
- `filelocation::String`: File directory (use backslashes)
- `filename::String`: Identifyer for the file (without sample index)
- `sample_idx::Int`: Sample index, to be added to the filename
- `num_plies::Int`: Number of plies
- `angles::Vector{<:Union{Int,Float64}}`: possible ply-angles in degrees
- `target_parameters::Vector{Float64}`: target lamination parameters
- `target_stack::Vector{Int}`: optimal solution, 
    producing the target lamination parameters
- `disorientation_constraint::Bool`: whether the disorientation constraint is used
- `c_strength::Float64`: strength of the constraint's penalty
- `c_distance::Union{Float64,Int}`: angle distance for the constraint
- `c_list::Union{Matrix{<:Int},BitMatrix}`: matrix of constraint violations for angle pairs,
    as generated with `disorientation_constraint_violations` in `generate_random_stacks.jl`
- `psi0_list::Vector{MPS}`: a vector of initial MPS for the different trials
- `psi0_energies_list::Vector{Float64}`: a list for the respective energy expectation values
- `sweeps::Sweeps`: the `Sweeps` object used in the algorithm
- `sweep_sequence::String`: the `sweep_sequence` used in the algorithm

# Returns
- `String`: filepath for the created file
"""
function create_file_for_dmrg_experiment(
    filelocation::String, filename::String, sample_idx::Int,
    num_plies::Int,angles::Vector{<:Union{Int,Float64}},
    target_parameters::Vector{Float64},target_stack::Vector{Int},
    disorientation_constraint::Bool,c_strength::Float64,
    c_distance::Union{Float64,Int},c_list::Union{Matrix{<:Int},BitMatrix},
    psi0_list::Vector{MPS},psi0_energies_list::Vector{Float64},sweeps::Sweeps,sweep_sequence::String
)
    # add backslash at the end of directory, if necessary
    if filelocation[end] ≠ '\\'
        filelocation *= "\\"
    end

    # create complete filepath
    filepath = filelocation*filename*"_sample_"*lpad(sample_idx,4,"0")*".hdf5"

    num_tries = length(psi0_list)
    num_sweeps = length(sweeps)

    # create HDF5 file
    fid = h5open(filepath,"cw")

    # store properties
    props = create_group(fid,"properties")
    attributes(props)["sample_idx"] = sample_idx
    attributes(props)["num_plies"] = num_plies
    attributes(props)["num_angles"] = length(angles)
    attributes(props)["sweep_sequence"] = sweep_sequence
    attributes(props)["num_sweeps"] = num_sweeps
    attributes(props)["num_tries"] = num_tries
    props["angles"] = angles
    props["target_parameters"] = target_parameters
    props["target_stack"] = target_stack
    attributes(props)["disorientation_constraint"] = disorientation_constraint
    attributes(props)["disorientation_constraint_distance"] = c_distance
    props["disorientation_constraint_matrix"] = c_list
    attributes(props)["disorientation_constraint_strength"] = c_strength

    group_sweeps = create_group(props,"sweeps")
    group_sweeps["maxdim"] = sweeps.maxdim
    group_sweeps["cutoff"] = sweeps.cutoff
    attributes(props)["maxdim"] = maximum(sweeps.maxdim)
    
    # store psi0
    group_psi0 = create_group(fid,"psi0")
    for (n,(psi0,psi0e)) ∈ enumerate(zip(psi0_list,psi0_energies_list))
        ITensors.HDF5.write(group_psi0,"try_$n",psi0)
        attributes(group_psi0["try_$n"])["energy"] = psi0e
    end

    # create datasets for data and results
    data = create_group(fid,"data")
    
    create_dataset(data,"energies",Float64,(num_tries,num_sweeps))
    create_dataset(data,"time_stamps",Float64,(num_tries,num_sweeps))
    create_dataset(data,"elapsed_time_sweeps",Float64,(num_tries,num_sweeps))
    create_dataset(data,"maxlinkdim",Int,(num_tries,num_sweeps))
    create_group(data,"psi_samples")

    res_group = create_group(fid,"results")
    create_dataset(res_group,"stack",Int,(num_tries,num_plies))
    create_dataset(res_group,"lamination_parameters",Float64,(num_tries,8))
    create_dataset(res_group,"loss",Float64,(num_tries,))
    create_dataset(res_group,"rmse",Float64,(num_tries,))
    create_dataset(res_group,"constraint_violations",Int,(num_tries,))
    create_group(res_group,"psi")

    close(fid)

    return filepath
end


"""
    gen_bond_dims(num_sweeps::Int,max_bond_dim::Int)

Generate list of maximum bond dimensions for the experiment.

The maximum bond dimension is kept constant for the majority of the optimization.
However, at the end, of the optimization, the bond dimension is divided in half
such that bond dimension is 1 for the last 2 sweeps.
"""
function gen_bond_dims(num_sweeps::Int,max_bond_dim::Int)
    decrease_bd = [max_bond_dim]
    while decrease_bd[end] > 1
        push!(decrease_bd,decrease_bd[end]÷2)
    end
    push!(decrease_bd,1)
    return append!(
        fill(max_bond_dim,num_sweeps-length(decrease_bd)),decrease_bd
    )
end

"""
    gen_cutoffs(num_sweeps::Int,co::Float64=1e-14)

Generate list of cutoffs for the singular values.

The cutoffs are kept constant at `co`, except for the last sweep, where the cutoff
is set to 0.5, to enforce a basis state.
"""
gen_cutoffs(num_sweeps::Int,co::Float64=1e-14) = append!(zeros(Float64,num_sweeps-1).+co,[0.5])

#%% calculate energy expectation for psi0 with Hlist

"""
    energy_Hsum_psi(Hlist::Vector{<:MPO},psi::MPS)

Calculate the energy expectation value of a sum of MPOs, stored as a vector
"""
energy_Hsum_psi(Hlist::Vector{<:MPO},psi::MPS) = sum([inner(psi',H,psi) for H ∈ Hlist])


"""
    energy_Hsum_psi(PH::ProjMPOSum,psi::MPS)

Calculate the energy expectation value of a sum of MPOs stored as a `ProjMPOSum`
"""
function energy_Hsum_psi(PH::ProjMPOSum,psi::MPS)
    terms = :terms ∈ fieldnames(ProjMPOSum) ? PH.terms : PH.pm
    return sum([inner(psi',p.H,psi) for p ∈ terms])
end

#%% dmrg experiment sweep

"""
    dmrg_experiment_one_try(
    num_plies::Int,
    PH::ProjMPOSum,psi0::MPS,
    sweeps::Sweeps,sweep_sequence::String;save_psi_freq::Int=10,
    kwargs...
)

Run DMRG for a single trial (for particular setting and a single initial MPS).

The function is a modified version of `dmrg_custom` from `dmrg_custom.jl`.

The modifications mainly consist of saving additional data during the optimization
and returning this data at the end of the function. Also, some settings were fixed,
that could be included as keyword arguments in `dmrg` (from ITensor) or `dmrg_costum`.

This function is used in `dmrg_experiment` which perform multiple trials with
different initial MPS psi0, and saves the gathered data into an HDF5 file.

# Arguments
- `num_plies::Int`: number of plies in the stack
- `PH::ProjMPOSum`: the Hamiltonians MPO
- `psi0::MPS`: the initial MPS
- `sweeps:Sweeps`: sweep setting (number of sweeps, max. bond dim., cutoffs, noise)
- `sweep_sequence::String`: sweep sequence, containing `R` for right-ward and
    `L` for left-ward sweeps
- `save_psi_freq::Int`: number of sweeps, after which the current MPS is sampled

# Keywords
- `kwargs...`: see `dmrg` (from ITensors.jl).
    The following keywords are already fixed and do not have an effect:
        svd_alg, obs, eigsolve_which_eigenvalue, write_when_maxdim_exceeds

# Returns
- `MPS`: final MPS
- `Vector{Float64}`: list of the recorded energy expectation value
    after each sweep
- `Vector{Float64}`: list of the recorded time stamps after each sweep
- `Vector{Float64}`: list of the durations of each sweep
- `Vector{Int}`: list of the maximum bond dimension in each sweep
- `Vector{MPS}`: list of MPS, the state saved after every saved_psi_list sweeps
"""
function dmrg_experiment_one_try(
    num_plies::Int,
    PH::ProjMPOSum,psi0::MPS,
    sweeps::Sweeps,sweep_sequence::String;save_psi_freq::Int=10,
    kwargs...
)
    # !!! change: fixed setting
    svd_alg = "divide_and_conquer"
    obs = NoObserver()
    write_when_maxdim_exceeds = nothing

    outputlevel::Int = get(kwargs, :outputlevel, 1)

    # eigsolve kwargs
    eigsolve_tol::Number = get(kwargs, :eigsolve_tol, 1e-14)
    eigsolve_krylovdim::Int = get(kwargs, :eigsolve_krylovdim, 3)
    eigsolve_maxiter::Int = get(kwargs, :eigsolve_maxiter, 1)
    eigsolve_verbosity::Int = get(kwargs, :eigsolve_verbosity, 0)

    ishermitian::Bool = get(kwargs, :ishermitian, true)

    eigsolve_which_eigenvalue::Symbol = :SR

    psi = copy(psi0)
    N = length(psi)

    # !!! added: copy PH
    PH = copy(PH)

    first_site = sweep_sequence[1] == 'L' ? length(psi) - 1 : 1 # orthogonalize to left of the two sites (-1)?
    if !isortho(psi) || ITensors.orthocenter(psi) != first_site
        orthogonalize!(psi, first_site)
    end
    @assert isortho(psi) && ITensors.orthocenter(psi) == first_site

    position!(PH, psi, first_site)
    energy = 0.0

    len_sweep_sequence = length(sweep_sequence)

    last_sweep_direction = 'X' # something different than 'L' and 'R', since already orthogonalized

    # !!! added: vectors for saving energies, durations,...
    num_sweeps = length(sweeps)
    energies_list = Vector{Float64}(undef,num_sweeps)
    times_list = Vector{Float64}(undef,num_sweeps)
    elapsed_times_sweeps = Vector{Float64}(undef,num_sweeps)
    maxlinkdim_list = Vector{Int}(undef,num_sweeps)
    num_psi_saved = num_sweeps ÷ save_psi_freq
    saved_psi_list = Vector{MPS}(undef,num_psi_saved)

    # !!! added: record time of sweeps
    t0 = time()
    for sw in 1:nsweep(sweeps)
        sweep_direction = sweep_sequence[(sw-1)%len_sweep_sequence+1]
        
        sw_time = @elapsed begin
            maxtruncerr = 0.0

            # !!! removed: writing for large maxdim

            first_site = sweep_direction == 'L' ? length(psi) - 1 : 1 # orthogonalize to left of the two sites
            if sweep_direction == last_sweep_direction
                orthogonalize!(psi, first_site)
            end
            
            iterator = sweep_direction == 'L' ? sweepnext_to_left(num_plies) : sweepnext_to_right(num_plies)
            for (b, ha) in iterator

                position!(PH, psi, b)

                phi = psi[b] * psi[b+1]

                vals, vecs = ITensors.eigsolve(
                    PH,
                    phi,
                    1,
                    eigsolve_which_eigenvalue;
                    ishermitian=ishermitian,
                    tol=eigsolve_tol,
                    krylovdim=eigsolve_krylovdim,
                    maxiter=eigsolve_maxiter
                )

                energy = vals[1]
                phi::ITensor = vecs[1]

                ortho = ha == 1 ? "left" : "right"

                drho = nothing
                if noise(sweeps, sw) > 0.0
                    drho = noise(sweeps, sw) * noiseterm(PH, phi, ortho)
                end


                spec = replacebond!(
                    psi,
                    b,
                    phi;
                    maxdim=maxdim(sweeps, sw),
                    mindim=mindim(sweeps, sw),
                    cutoff=cutoff(sweeps, sw),
                    eigen_perturbation=drho,
                    ortho=ortho,
                    normalize=true,
                    svd_alg=svd_alg
                )
                maxtruncerr = max(maxtruncerr, spec.truncerr)

                if outputlevel >= 2
                    @printf("Sweep %d, half %d, bond (%d,%d) energy=%s\n", sw, ha, b, b + 1, energy)
                    @printf(
                        "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
                        cutoff(sweeps, sw),
                        maxdim(sweeps, sw),
                        mindim(sweeps, sw)
                    )
                    @printf(
                        "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
                    )
                    flush(stdout)
                end

                if sweep_direction == 'R'
                    sweep_is_done = (b == N-1 && ha == 1)
                else
                    sweep_is_done = (b == 1 && ha == 2)
                end
                measure!(
                    obs;
                    energy=energy,
                    psi=psi,
                    bond=b,
                    sweep=sw,
                    half_sweep=ha,
                    spec=spec,
                    outputlevel=outputlevel,
                    sweep_is_done=sweep_is_done
                )
            end
        end

        # !!! add: record time stamp after sweep
        t1 = time() - t0

        if outputlevel >= 1
            @printf(
                "After sweep %d %s energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
                sw,
                sweep_direction,
                energy,
                maxlinkdim(psi),
                maxtruncerr,
                sw_time
            )
            flush(stdout)
        end

        # !!! add: save energy and duration in arrays
        energies_list[sw] = energy
        times_list[sw] = t1
        elapsed_times_sweeps[sw] = sw_time
        maxlinkdim_list[sw] = maxlinkdim(psi)
        if sw % save_psi_freq == 0
            saved_psi_list[sw÷save_psi_freq] = copy(psi)
        end
        
        last_sweep_direction = sweep_direction
        
        isdone = checkdone!(obs; energy=energy, psi=psi, sweep=sw, outputlevel=outputlevel)
        isdone && break
    end

    return psi,energies_list,times_list,elapsed_times_sweeps,maxlinkdim_list,saved_psi_list

end


"""
    count_disorientation_constraint_violations(stack::Vector{Int},c_list::Matrix{<:Int})

Count constraint violations
"""
count_disorientation_constraint_violations(stack::Vector{Int},c_list::Matrix{<:Int}) = sum([
    c_list[stack[n],stack[n+1]] for n ∈ 1:(length(stack)-1)
])


"""
    create_samples(psi::MPS,num_samples::Int)

Create num_samples from the superposition in state psi
"""
function create_samples(psi::MPS,num_samples::Int)
    sample_list = Matrix{Int}(undef,length(psi),num_samples)
    orthogonalize!(psi,1)
    for k ∈ 1:num_samples
        sample_list[:,k] = sample(psi)
    end
    return sample_list
end

"""
    result_psi_to_stack(psi::MPS)

Convert the MPS psi after the last sweep into a stacking sequence
"""
function result_psi_to_stack(psi::MPS)
    stack = Vector{Int}(undef,length(psi))
    sites = siteinds(psi)
    for (n,t) ∈ enumerate(psi)
        if n == 1
            bi = commonind(t,psi[n+1])
            stack[n] = argmax([t[bi=>1,sites[n]=>s]^2 for s ∈ 1:dim(sites[n])])
            continue
        elseif n == length(psi)
            bi = commonind(t,psi[n-1])
            stack[n] = argmax([t[bi=>1,sites[n]=>s]^2 for s ∈ 1:dim(sites[n])])
            continue
        end
        bi1 = commonind(t,psi[n-1])
        bi2 = commonind(t,psi[n+1])
        stack[n] = argmax([t[bi1=>1,bi2=>1,sites[n]=>s]^2 for s ∈ 1:dim(sites[n])])
    end
    return stack
end


"""
    dmrg_experiment(
    filelocation::String, filename::String, sample_idx::Int, num_tries::Int,
    angles::Vector{<:Union{Int,Float64}},target_parameters::Vector{Float64},
    target_stack::Vector{Int},disorientation_constraint::Bool,c_strength::Float64,
    c_distance::Union{Float64,Int},c_list::Union{Matrix{<:Int},BitMatrix},
    sites::Vector{<:Index}, PH::ProjMPOSum,psi0_list::Vector{MPS},
    num_sweeps::Int, sweep_sequence::String, max_bond_dim::Int;
    psi_num_samples::Int=100,save_psi_freq::Int=10,
    kwargs...
)

Run DRMG `num_tries` times and save results.

This function runs the experiment!
It creates the file with `create_file_for_dmrg_experiment` and performs
`num_tries` trails with the same settings, except for different initial
MPS with `dmrg_experiment_one_try`. Finally, it stores the recorded data
and results in the file.

# Arguments
- `filelocation::String`: file directory for the HDF5 file
- `filename::String`: identifyer for the file name
- `sample_idx::Int`: sample index of the target parameters,
    is added to the filename
- `num_tries::Int`: number of tries for a specific setting
- `angles::Vector{<:Union{Int,Float64}}`: possible ply angles in degrees
- `target_parameters::Vector{Float64}`: target lamination parameters
- `target_stack::Vector{Int}`: solution corresponding to the target parameters
- `disorientation_constraint::Bool`: whether to include the disorientation constraint
- `c_strength::Float64`: strength of the penalty for the constraint
- `c_distance::Union{Float64,Int}`: angle distance for the disorientation constraint
- `c_list::Union{Matrix{<:Int},BitMatrix}`: matrix of constraint violations for angle pairs,
    as generated with `disorientation_constraint_violations` in `generate_random_stacks.jl`
- `sites::Vector{<:Index}`: indices used in the Hamiltonian and MPS
- `PH::ProjMPOSum`: `ProjMPOSum` object for the Hamiltonian
- `psi0_list::Vector{MPS}`: list of initial MPS, has length `num_tries`
- `num_sweeps::Int`: number of sweeps
- `sweep_sequence::String`: sweep sequence, 
    as specified in `dmrg_costum`/`dmrg_experiment_one_try`
- `max_bond_dim::Int`: maximum bond dimension in the optimization

# Keywords:
- `psi_num_samples::Int=100`: number of generated stacking sequences
    from the superposition in the MPS during the optimization
- `save_psi_freq::Int=10`: the samples from the MPS are generated every
    `save_psi_freq` sweeps
- `kwargs...`: further keywords for `dmrg_experiment_one_try`
"""
function dmrg_experiment(
    filelocation::String, filename::String, sample_idx::Int, num_tries::Int,
    angles::Vector{<:Union{Int,Float64}},target_parameters::Vector{Float64},
    target_stack::Vector{Int},disorientation_constraint::Bool,c_strength::Float64,
    c_distance::Union{Float64,Int},c_list::Union{Matrix{<:Int},BitMatrix},
    sites::Vector{<:Index}, PH::ProjMPOSum,psi0_list::Vector{MPS},
    num_sweeps::Int, sweep_sequence::String, max_bond_dim::Int;
    psi_num_samples::Int=100,save_psi_freq::Int=10,
    kwargs...
)
    num_plies = length(sites)

    # calculate energies for initial MPS
    psi0_energies_list = [energy_Hsum_psi(PH,psi0) for psi0 ∈ psi0_list]

    # create sweeps
    bond_dims = gen_bond_dims(num_sweeps,max_bond_dim)
    cutoffs = gen_cutoffs(num_sweeps)
    sweeps = Sweeps(num_sweeps)
    maxdim!(sweeps,bond_dims...)
    cutoff!(sweeps,cutoffs...)


    # create file
    filepath = create_file_for_dmrg_experiment(
        filelocation, filename, sample_idx,
        num_plies,angles,
        target_parameters,target_stack,disorientation_constraint,
        c_strength,c_distance,c_list,
        psi0_list,psi0_energies_list,sweeps,sweep_sequence
    )

    # perform optimizations
    for (t,psi0) ∈ enumerate(psi0_list)
        # output
        println("Try $t:")
        println("")

        # perform dmrg
        psi,energies_list,times_list,elapsed_time_sweeps,maxlinkdim_list,saved_psi_list = dmrg_experiment_one_try(
            num_plies,PH,psi0,sweeps,sweep_sequence;save_psi_freq=save_psi_freq,kwargs...
        )

        # get resulting stacking sequence, lamination parameters,
        # loss and constraint violations
        stack = result_psi_to_stack(psi)
        res_lp = lamination_parameters(angles[stack])
        res_loss = sum((res_lp .- target_parameters).^2)
        c_violations = count_disorientation_constraint_violations(stack,c_list)

        # print results
        println("Completed!")
        println("Last time:     $(times_list[end])")
        println("Last energy:   $(energies_list[end])")
        println("Loss:          $(res_loss)")
        println("RMSE:          $(sqrt(res_loss))")
        println("Constr. viol.: $(c_violations)")
        println("✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳")
        println("")

        # create samples from stored intermediate MPS
        println("Creating samples...")
        psi_samples = [create_samples(p,psi_num_samples) for p ∈ saved_psi_list]
        
        # Save results in HDF5
        # ❗ mistake: c_violations is not saved in the HDF5
        println("Saving...")
        fid = h5open(filepath,"r+")
        fid["data/energies"][t,:] = energies_list
        fid["data/time_stamps"][t,:] = times_list
        fid["data/elapsed_time_sweeps"][t,:] = elapsed_time_sweeps
        fid["data/maxlinkdim"][t,:] = maxlinkdim_list
        
        res_group = fid["results"]
        ITensors.HDF5.write(res_group["psi"],"try_$t",psi)
        res_group["stack"][t,:] = stack
        res_group["lamination_parameters"][t,:] = res_lp
        res_group["loss"][t] = res_loss
        res_group["rmse"][t] = sqrt(res_loss)

        psi_samples_group = create_group(fid["data/psi_samples"],"try_$t")
        for (k,sample_list) ∈ enumerate(psi_samples)
            psi_samples_group["sweep_$(k*save_psi_freq)"] = sample_list
        end
        close(fid)
        println("Saved!")
        println("")

    end
end


"""
    dmrg_experiment(
    filelocation::String, filename::String, sample_idx::Int, num_tries::Int,
    angles::Vector{<:Union{Int,Float64}},target_parameters::Vector{Float64},
    target_stack::Vector{Int},disorientation_constraint::Bool,c_strength::Float64,
    c_distance::Union{Float64,Int},c_list::Union{Matrix{<:Int},BitMatrix},
    sites::Vector{<:Index}, Hlist::Vector{<:MPO},psi0_list::Vector{MPS},
    num_sweeps::Int, sweep_sequence::String, max_bond_dim::Int;
    psi_num_samples::Int=100,save_psi_freq::Int=10,
    kwargs...
)

Wrapper for `dmrg_experiment(..., PH::ProjMPOSum)` for a sum of MPO, given
as a vector `Hlist::Vector{<:MPO}` 
"""
function dmrg_experiment(
    filelocation::String, filename::String, sample_idx::Int, num_tries::Int,
    angles::Vector{<:Union{Int,Float64}},target_parameters::Vector{Float64},
    target_stack::Vector{Int},disorientation_constraint::Bool,c_strength::Float64,
    c_distance::Union{Float64,Int},c_list::Union{Matrix{<:Int},BitMatrix},
    sites::Vector{<:Index}, Hlist::Vector{<:MPO},psi0_list::Vector{MPS},
    num_sweeps::Int, sweep_sequence::String, max_bond_dim::Int;
    psi_num_samples::Int=100,save_psi_freq::Int=10,
    kwargs...
)
    Hlist .= permute.(Hlist, Ref((linkind, siteinds, linkind)))
    PH = ProjMPOSum(Hlist)
    return dmrg_experiment(
        filelocation, filename, sample_idx, num_tries,
        angles,target_parameters,target_stack,disorientation_constraint,c_strength,c_distance,c_list,
        sites, PH,psi0_list,num_sweeps, sweep_sequence, max_bond_dim;
        psi_num_samples=psi_num_samples,save_psi_freq=save_psi_freq,
        kwargs...
    )
end

"""
    dmrg_experiment(
    filelocation::String, filename::String, sample_idx::Int, num_tries::Int,
    angles::Vector{<:Union{Int,Float64}},target_parameters::Vector{Float64},
    target_stack::Vector{Int},disorientation_constraint::Bool,
    c_strength::Float64,c_distance::Union{Float64,Int},c_list::Union{Matrix{<:Int},BitMatrix},
    sites::Vector{<:Index}, H::MPO,psi0_list::Vector{MPS},
    num_sweeps::Int, sweep_sequence::String, max_bond_dim::Int;
    psi_num_samples::Int=100,save_psi_freq::Int=10,
    kwargs...
)

Wrapper for `dmrg_experiment(..., PH::ProjMPOSum)` for a sum of MPO, given
as a vector `H::MPO`
"""
function dmrg_experiment(
    filelocation::String, filename::String, sample_idx::Int, num_tries::Int,
    angles::Vector{<:Union{Int,Float64}},target_parameters::Vector{Float64},
    target_stack::Vector{Int},disorientation_constraint::Bool,
    c_strength::Float64,c_distance::Union{Float64,Int},c_list::Union{Matrix{<:Int},BitMatrix},
    sites::Vector{<:Index}, H::MPO,psi0_list::Vector{MPS},
    num_sweeps::Int, sweep_sequence::String, max_bond_dim::Int;
    psi_num_samples::Int=100,save_psi_freq::Int=10,
    kwargs...
)
    return dmrg_experiment(
        filelocation, filename, sample_idx, num_tries,
        angles,target_parameters,target_stack,disorientation_constraint,c_strength,c_distance,c_list,
        sites,[H],psi0_list,num_sweeps, sweep_sequence, max_bond_dim; 
        psi_num_samples=psi_num_samples,save_psi_freq=save_psi_freq,
        kwargs...
    )
end