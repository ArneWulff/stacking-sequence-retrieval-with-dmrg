"""
File to run the experiments

This file shows to perform the DMRG experiments like we did in the
paper in section 5. It shows how to include the relevant files, and in which order to
run the different functions.
"""

# important imports
using Printf
using HDF5
using DataFrames

using ITensors
using Random: randstring

# include functions
include("laminationparameters.jl")
include("mpo.jl")
include("dmrg_custom.jl")
include("dmrg_experiment.jl")


let
    # filepath of HDF5 file for target lamination parameters
    filepath_lp_sample = "..."

    # filepath and identifier for data files from experiments
    filelocation = "..."
    filename = "constant_bondsize"

    # file location and identifier for dummy file
    filelocation_dummy = "..."
    filename_dummy = "dummy"

    # general settings
    angles = [0, 45, 90, -45]
    num_plies = 200
    num_angles = length(angles)
    num_tries = 5
    psi_num_samples = 100
    save_psi_freq = 10

    # constraint settings
    disorientation_constraint = false # or true
    angle_constraint_distance = 45 + 1
    penalty_per_violation = 0.005

    # generate matrix for constraint violations
    c_list = Int.(generate_disorientation_constraint_pq_list(angles,angle_constraint_distance,1.0)[2])

    # starting time stamp
    t0 = time()

    # Sample indices for target lamination parameters,
    # for which the optimization is performed.
    # We used the first 50 sets of target lamination prameters from the generated file
    # (which includes 100 sets of lamination parameters)
    sample_idx_list = Vector(0:49)

    # settings for optimization
    num_sweeps_list = [69]
    max_bond_dim_list = [2,4,8,16,32]
    sweep_sequence_list = ["L","R","LR"]

    for (nsid,sample_idx) ∈ enumerate(sample_idx_list)
        println("Sample $nsid with index $sample_idx")
        println("--------------------------")
        println("")

        # get target parameters and stack
        fid_lp = h5open(filepath_lp_sample,"r")
        target_parameters = fid_lp["sample/lamination_parameters"][:,(sample_idx+1)]
        target_stack = fid_lp["sample/stacking_sequence"][:,(sample_idx+1)]
        close(fid_lp)

        # build Hamiltonian and sites
        sites = siteinds("Qudit",num_plies;dim=num_angles)
        trig_array = generate_trig_array(angles)
        weights_d = generate_weights_d(num_plies)
        ev_array = generate_eigenval_array(trig_array, weights_d, target_parameters,distribute_target="proportion")
        Hlist = MPO[]
        for X in 1:2
            for l in 1:4
                Hmpo = build_partial_mpo(X,l,sites,ev_array)
                push!(Hlist,Hmpo)
            end
        end

        # add penalty
        if disorientation_constraint
            p_list,q_list = generate_disorientation_constraint_pq_list(angles,angle_constraint_distance,penalty_per_violation)
            Hpenalty = build_mpo_disorientation_constraint(sites,p_list,q_list)
            push!(Hlist,Hpenalty)
        end

        println("Building Hamiltonian complete.")
        println("")

        # dummy run to compile functions
        if nsid == 1
            println("Do dummy run to compile functions")
            println("")
            if !isdir(filelocation_dummy)
                mkpath(filelocation_dummy)
            end
            rs = randstring(10)
            while isfile(filelocation_dummy*filename_dummy*"_"*rs*".hdf5")
                rs = randstring(10)
            end
            filename_dummy_rs = filename_dummy*"_"*rs*".hdf5"
            # start with linkdim 2
            psi0_list = [randomMPS(sites,linkdims=2) for k ∈ 1:1]
            dmrg_experiment(
                filelocation_dummy, filename_dummy_rs, sample_idx, 1,
                angles,target_parameters,target_stack,
                disorientation_constraint,penalty_per_violation,
                angle_constraint_distance,c_list,
                sites, Hlist,psi0_list,
                10, "L", 2
            )
            println("")
            println("Dummy run completed!")
            println("⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆")
            println("")
        end

        # run experiments
        for num_sweeps ∈ num_sweeps_list, max_bond_dim ∈ max_bond_dim_list, sweep_sequence ∈ sweep_sequence_list
            println("num_sweeps $num_sweeps, max_bond_dim $max_bond_dim, sweep_sequence $sweep_sequence")
            println("")
            filelocation_run = filelocation
            filelocation_run *= disorientation_constraint ? "with_constraint\\" : "without_constraint\\"
            filelocation_run *= "num_sweeps_$(num_sweeps)\\max_bond_dim_$(max_bond_dim)\\"
            filelocation_run *= "sweep_$(sweep_sequence)\\"
            if !isdir(filelocation_run)
                mkpath(filelocation_run)
            end
            filelocation_for_psi0 = filelocation * "without_constraint\\num_sweeps_$(num_sweeps)\\max_bond_dim_2\\sweep_L\\"
            filelocation_for_psi0 *= filename*"_sample_"*lpad(sample_idx,4,"0")*".hdf5"
            if isfile(
                filelocation_for_psi0
            ) && (max_bond_dim > 2 || sweep_sequence != "L" || num_sweeps > 20 || disorientation_constraint)
                fid = h5open(filelocation_for_psi0,"r")
                if read(attributes(fid["properties"])["num_tries"]) ≥ num_tries
                    psi0_list = [ITensors.HDF5.read(fid["psi0"],"try_$k",MPS) for k ∈ 1:num_tries]
                    close(fid)
                    psi0_list = [replace_siteinds!(p, sites) for p ∈ psi0_list]
                else
                    close(fid)
                    psi0_list = [randomMPS(sites,2) for k ∈ 1:num_tries]
                end
            else
                psi0_list = [randomMPS(sites,2) for k ∈ 1:num_tries]
            end
            dmrg_experiment(
                filelocation_run, filename, sample_idx, num_tries,
                angles,target_parameters,target_stack,
                disorientation_constraint,penalty_per_violation,
                angle_constraint_distance,c_list,
                sites, Hlist,psi0_list,
                num_sweeps, sweep_sequence, max_bond_dim;
                psi_num_samples=psi_num_samples,save_psi_freq=save_psi_freq
            )
            println("")
            println("⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆")
            println("")
        end
        println("")
        println("Completed sample $nsid with index $sample_idx.")
        t1 = time() - t0
        h = lpad(floor(Int,t1/3600),2,'0')
        m = lpad(floor(Int,(t1 % 3600)/60),2,'0')
        s = lpad(floor(Int,(t1 % 60)),2,'0')
        ms = "$(t1 % 1)"[3:4]
        println("Time: $h:$m:$s.$ms")
        println("")
        println("♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦")
        println("")
    end

    println("")
    println("Finished!")    
    t1 = time() - t0
    h = lpad(floor(Int,t1/3600),2,'0')
    m = lpad(floor(Int,(t1 % 3600)/60),2,'0')
    s = lpad(floor(Int,(t1 % 60)),2,'0')
    ms = "$(t1 % 1)"[3:4]
    println("Time: $h:$m:$s.$ms")





end





