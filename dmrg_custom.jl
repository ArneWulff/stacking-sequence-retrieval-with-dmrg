"""
Modified DMRG implementation, that supports custom
sweeping directions

In the employed version v0.3.22 of ITensors.jl, the `dmrg` function
only supports alternating sweeps. One right-ward and one left-ward 
sweep are counted together as one sweep.

Here, we modify the DMRG-implementation to support any sequence of
sweeping directions. Consequently, we count a single right-ward sweep
or a single left-ward sweep as one sweep.

**Note:** The function `dmrg_custom` is not directly used in our experiments
but exemplifies the necessary changes. Our experiments use functions 
with additional modification for extracting additional information and
saving data in our chosen file-format (see function `dmrg_experiment_one_try`
in `dmrg_emperiment.jl`).
"""

"""
    SweepNextToRight

Variant of `SweepNext` (in ITensors.jl: src/mps/sweeps.jl), but explicitly used for
right-ward sweeps in `Base.iterate`.
"""
struct SweepNextToRight
  N::Int
  ncenter::Int
end


"""
    sweepnext_to_right(N::Int; ncenter::Int=2)::SweepNextToRight

Variant of `sweepnext(N::Int; ncenter::Int=2)` (in ITensors.jl: src/mps/sweeps.jl),
but returns a SweepNextToRight
"""
function sweepnext_to_right(N::Int; ncenter::Int=2)::SweepNextToRight
  if ncenter < 0
    error("ncenter must be non-negative")
  end
  return SweepNextToRight(N, ncenter)
end


"""
    Base.iterate(sn::SweepNextToRight, state=(0, 1))

Variant of `Base.iterate(sn::SweepNext, state=(0, 1))` 
(in ITensors.jl: src/mps/sweeps.jl), but only performing a right-ward sweep.

Produces states (1, 1), (2, 1), ..., (N-1, 1)
"""
function Base.iterate(sn::SweepNextToRight, state=(0, 1))
  b, ha = state
  bstop = sn.N - sn.ncenter + 2
  new_b = b + 1
  new_ha = ha
  done = false
  if new_b == bstop
    return nothing
  end
  return ((new_b, new_ha), (new_b, new_ha))
end

"""
    SweepNextToLeft

Variant of `SweepNext` (in ITensors.jl: src/mps/sweeps.jl), but explicitly used for
left-ward sweeps in `Base.iterate`.
"""
struct SweepNextToLeft
  N::Int
  ncenter::Int
end

"""
    sweepnext_to_left(N::Int; ncenter::Int=2)::SweepNextToLeft

Variant of `sweepnext(N::Int; ncenter::Int=2)` (in ITensors.jl: src/mps/sweeps.jl),
but returns a SweepNextToLeft
"""
function sweepnext_to_left(N::Int; ncenter::Int=2)::SweepNextToLeft
  if ncenter < 0
    error("ncenter must be non-negative")
  end
  return SweepNextToLeft(N, ncenter)
end

"""
    Base.iterate(sn::SweepNextToLeft, state=(0, 1))

Variant of `Base.iterate(sn::SweepNext, state=(0, 1))` 
(in ITensors.jl: src/mps/sweeps.jl), but only performing a left-ward sweep

Produces states (N-1, 2),(N-2, 2), ..., (1, 2)
"""
function Base.iterate(sn::SweepNextToLeft, state=(-1, 2))
  b, ha = state
  if b == -1
    b = sn.N - sn.ncenter + 2
  end
  bstop = 0
  new_b = b - 1
  new_ha = ha
  done = false
  if new_b == bstop
    return nothing
  end
  return ((new_b, new_ha), (new_b, new_ha))
end

#%%




#%% dmrg with custom sweep sequence

"""
    dmrg_custom(PH::ProjMPOSum, psi0::MPS, sweeps::Sweeps; sweep_sequence="LR", print_state=false, kwargs...)

Variant of `dmrg(PH, psi0::MPS, sweeps::Sweeps; kwargs...)` (in ITensors.jl: src/mps/dmrg.jl)
but with support for custom sequences of sweep directions.

The main difference is the inclusion of the `sweep_sequence` argument. 
`R` denotes a right-ward sweep (from 1 to N), `L` denotes a left-ward sweep.
If `length(sweep_direction)<length(sweeps)`, then `sweep_direction` is repeated.
Thus, `LR` will produce alternating sweeps, starting with a left-ward sweep.
`L` will produce only left-wards sweeps, and `R` will produce only right-ward sweeps.
However, more complicated sweep sequences like `LLRRLRLR` are also possible.

Note: The orginal `dmrg` function produces double the number of sweeps, since one
right-ward and one left-ward sweep are counted together as one sweep.

Note: debug checks and times in the code were ommited

# Arguments
- `PH::ProjMPOSum`: Hamiltonian in ProjMPOSum form (in ITensors.jl: src/mps/projmposum.jl)
- `psi0::MPS`: initial MPS
- `sweeps::Sweeps`: specifying number of sweeps, truncation, noise...
- `sweep_sequence::String`: Sweep sequence (as described above)

# Keywords
- `kwargs...`: see original `dmrg` function

"""
function dmrg_custom(PH::ProjMPOSum, psi0::MPS, sweeps::Sweeps; sweep_sequence::String="LR", kwargs...)
  if length(psi0) == 1
    error(
      "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
    )
  end

  # !!! removed @debug_check

  which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, nothing)
  svd_alg::String = get(kwargs, :svd_alg, "divide_and_conquer")
  obs = get(kwargs, :observer, NoObserver())
  outputlevel::Int = get(kwargs, :outputlevel, 1)

  write_when_maxdim_exceeds::Union{Int,Nothing} = get(
    kwargs, :write_when_maxdim_exceeds, nothing
  )
  write_path = get(kwargs, :write_path, tempdir())

  # eigsolve kwargs
  eigsolve_tol::Number = get(kwargs, :eigsolve_tol, 1e-14)
  eigsolve_krylovdim::Int = get(kwargs, :eigsolve_krylovdim, 3)
  eigsolve_maxiter::Int = get(kwargs, :eigsolve_maxiter, 1)
  eigsolve_verbosity::Int = get(kwargs, :eigsolve_verbosity, 0)

  ishermitian::Bool = get(kwargs, :ishermitian, true)

  eigsolve_which_eigenvalue::Symbol = :SR
  
  # !!! removed keyword argument deprecations

  psi = copy(psi0)
  N = length(psi)
  
  # !!! change: set first site and orthogonal center 
  #             according to sweep direction
  first_site = sweep_sequence[1] == 'L' ? length(psi) - 1 : 1 
  if !isortho(psi) || ITensors.orthocenter(psi) != first_site
    orthogonalize!(psi, first_site)
  end
  @assert isortho(psi) && ITensors.orthocenter(psi) == first_site

  
  # !!! change: set position of PH to according first site
  position!(PH, psi, first_site)
  energy = 0.0
 
  # !!! added: variable for length of sweep_sequence
  len_sweep_sequence = length(sweep_sequence)
  
  # !!! added: variable for last sweep direction, used to
  #            determine, if orthogonalization is needed
  last_sweep_direction = 'X' # something different than 'L' and 'R', since already orthogonalized

  for sw in 1:nsweep(sweeps)
    # !!! added: get next sweep direction
    sweep_direction = sweep_sequence[(sw-1)%len_sweep_sequence+1]

    sw_time = @elapsed begin
      maxtruncerr = 0.0

      if !isnothing(write_when_maxdim_exceeds) &&
         maxdim(sweeps, sw) > write_when_maxdim_exceeds
        if outputlevel >= 2
          println(
            "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and
            maxdim(sweeps, sw) = $(maxdim(sweeps, sw)), writing environment tensors to disk"
          )
        end
        PH = disk(PH; path=write_path)
      end

      # !!! added: If last sweep direction == this sweep direction,
      #            orthogonalize to according site
      first_site = sweep_direction == 'L' ? length(psi) - 1 : 1 # orthogonalize to left of the two sites
      if sweep_direction == last_sweep_direction
        orthogonalize!(psi, first_site)
      end


      # !!! change: get according iterator
      iterator = sweep_direction == 'L' ? sweepnext_to_left(N) : sweepnext_to_right(N)
      for (b, ha) in iterator
        
        # !!! removed: debug checks, debug timers

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
          which_decomp=which_decomp,
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
        
        # !!! change: adjust sweep_is_done
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

    # !!! change: add sweep_direction to output
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

    # !!! add: update last_sweep_direction
    last_sweep_direction = sweep_direction

    isdone = checkdone!(obs; energy=energy, psi=psi, sweep=sw, outputlevel=outputlevel)
    isdone && break
  end
  return (energy, psi)
end


"""
    dmrg_custom(Hlist::Vector{<:MPO}, psi0::MPS, sweeps::Sweeps; sweep_sequence::String="LR", kwargs...)

Wrapper for `dmrg_custom(PH::ProjSumMPO,...)` for sums of MPO
"""
function dmrg_custom(Hlist::Vector{<:MPO}, psi0::MPS, sweeps::Sweeps; sweep_sequence::String="LR", kwargs...)
    Hlist .= permute.(Hlist, Ref((linkind, siteinds, linkind)))
    PH = ProjMPOSum(Hlist)
    return dmrg_custom(PH,psi0,sweeps;sweep_sequence=sweep_sequence,kwargs...)
end


"""
    dmrg_custom(H::MPO,psi0::MPS, sweeps::Sweeps; sweep_sequence::String="LR", kwargs...)

    Wrapper for `dmrg_custom(PH::ProjSumMPO,...)` for a single MPO
"""
function dmrg_custom(H::MPO,psi0::MPS, sweeps::Sweeps; sweep_sequence::String="LR", kwargs...)
    return dmrg_custom([H],psi0,sweeps;sweep_sequence=sweep_sequence,kwargs...)
end