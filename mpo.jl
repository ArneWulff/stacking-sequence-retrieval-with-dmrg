""" 
Functions to build the MPO for the stacking sequence retrieval problem.
See section 4 of the paper.
"""


"""
    generate_trig_array(angles::Union{Vector{Float64},Vector{Int}})

Calculate the results of the trigonometric functions corresonding to the
individual lamination parameters for the different possible angles.

# Arguments
- `angles::Union{Vector{Float64},Vector{Int}}`: possible ply angles, in degrees

# Returns
- `Matrix{Float64}`: Array with fₗ(s) at position (s,l),
                     where s is the index of ply angle θₛ and l the index of the
                     lamination parameter.
                     f₁(s) = cos(2θₛ), f₂(s) = sin(2θₛ), f₃(s) = cos(4θₛ), f₄(s) = sin(4θₛ)
"""
function generate_trig_array(angles::Union{Vector{Float64},Vector{Int}})
  num_angles = length(angles)
  arr = zeros((num_angles, 4))
  for a in 1:num_angles
    arr[:, 1] .= cospi.(2 * angles / 180)
    arr[:, 2] .= sinpi.(2 * angles / 180)
    arr[:, 3] .= cospi.(4 * angles / 180)
    arr[:, 4] .= sinpi.(4 * angles / 180)
  end
  return arr
end;


"""
    generate_weights_d(num_plies::Int)

Calculate weights for D-lamination parameters for a symmetric laminate.

# Arguments
- `num_plies::Int`: number of plies

# Returns
- `Vector{Float64}`: Weights for the D-lamination parameters.
                     αᴰₙ = ((zₙ)³-(zₙ₋₁)³)/N³,  zₙ = n
"""
function generate_weights_d(num_plies::Int)
  boundaries_3 = (range(0, num_plies) / num_plies) .^ 3
  return boundaries_3[2:end] - boundaries_3[1:end-1]
end;


"""
    generate_eigenval_array(trig_array::Matrix{Float64},
      weights::Vector{Float64},
      target::Vector{Float64};
      distribute_target::String="even")

Generate the eigenvalues hₛ⁽ⁿ⁾ of the local operators ̂H⁽ⁿ⁾

# Arguments
- `trig_array::Matrix{Float64}`: array from `generate_trig_array`
- `weights::Vector{Float64}`: weights from `generate_weights_d`
- `target::Vector{Float64}`: target lamination parameters

# Keywords
- `distribute_target::String="proportion"`: how to integrate the target parameters into
      the local operators. Options:
      + `"proportion"`: according to the weights of the lamination parameters (as described in the paper)
      + `"even"`: evenly accross all terms

# Returns
- `Array{Float64,4}`: Array with the eigenvalues of shape (2,4,num_plies,num_angles).
                      The first index represent the A or D lamination parameters,
                      the second index which one of the 4 parameters for A or D.
"""
function generate_eigenval_array(trig_array::Matrix{Float64},
    weights::Vector{Float64},
    target::Vector{Float64};
    distribute_target::String="proportion")
  @assert distribute_target ∈ ["even","proportion"]
  num_angles = size(trig_array, 1)
  num_plies = length(weights)
  arr = zeros((2, 4, num_plies, num_angles))
  for a in 1:num_angles
    for n in 1:num_plies
      for l in 1:4
        arr[1, l, n, a] = (trig_array[a, l] - target[l]) / num_plies
        if distribute_target == "even"
          arr[2, l, n, a] = trig_array[a, l] * weights[n] - target[l+4] / num_plies
        else
          arr[2, l, n, a] = (trig_array[a, l] - target[l+4]) * weights[n]
        end
      end
    end
  end
  return arr
end;
#%%

#%% build tensor, middle
"""
    build_mpo_tensor_middle(X::Int, l::Int, n::Int, ev_array::Array{Float64,4},
        lefti::Index, righti::Index, upi::Index, downi::Index)

Build tensor for MPO, sites n = 2,...,num_plies-1

# Arguments
- `X::Int`: Select A or D parameters, `X=1` for A and `X=2` for D
- `l::Int`: Which one of the four parameters, l ∈ {1,2,3,4}
- `n::Int`: site index, n ∈ {2,...,num_plies-1}
- `ev_array::Array{Float64,4}`: eigenvalue array from `generate_eigenval_array`
- `lefti::Index`: left bond index
- `righti::Index`: right bond index
- `upi::Index`: site index
- `downi::Index`: primed site index

# Returns
- `ITensor`: local MPO tensor for site `n`
"""
function build_mpo_tensor_middle(X::Int, l::Int, n::Int, ev_array::Array{Float64,4},
    lefti::Index, righti::Index, upi::Index, downi::Index)
  num_angles = size(ev_array, 4)
  T = ITensor(lefti, righti, upi, downi)
  T .= 0.
  for a ∈ 1:num_angles
    h = ev_array[X, l, n, a]
    T[lefti=>1, righti=>1, upi=>a, downi=>a] = 1.
    T[lefti=>2, righti=>2, upi=>a, downi=>a] = 1.
    T[lefti=>3, righti=>3, upi=>a, downi=>a] = 1.
    T[lefti=>1, righti=>2, upi=>a, downi=>a] = sqrt(2)*h
    T[lefti=>2, righti=>3, upi=>a, downi=>a] = sqrt(2)*h
    T[lefti=>1, righti=>3, upi=>a, downi=>a] = h^2
  end
  return T
end;
#%%

#%% build tensor, begin
"""
    build_mpo_tensor_begin(X::Int, l::Int, ev_array::Array{Float64,4},
      righti::Index, upi::Index, downi::Index)

Build tensor for MPO at site 1

# Arguments
- `X::Int`: Select A or D parameters, `X=1` for A and `X=2` for D
- `l::Int`: Which one of the four parameters, l ∈ {1,2,3,4}
- `ev_array::Array{Float64,4}`: eigenvalue array from `generate_eigenval_array`
- `righti::Index`: right bond index
- `upi::Index`: site index
- `downi::Index`: primed site index

# Returns
- `ITensor`: local MPO tensor for site 1
"""
function build_mpo_tensor_begin(X::Int, l::Int, ev_array::Array{Float64,4},
    righti::Index, upi::Index, downi::Index)
  num_angles = size(ev_array, 4)
  T = ITensor(righti, upi, downi)
  T .= 0.
  for a ∈ 1:num_angles
    h = ev_array[X, l, 1, a]
    T[righti=>1, upi=>a, downi=>a] = 1.
    T[righti=>2, upi=>a, downi=>a] = sqrt(2)*h
    T[righti=>3, upi=>a, downi=>a] = h^2
  end
  return T
end;

#%% build tensor, end
"""
    build_mpo_tensor_end(X::Int, l::Int, ev_array::Array{Float64,4},
        lefti::Index, upi::Index, downi::Index)

# Arguments
- `X::Int`: Select A or D parameters, `X=1` for A and `X=2` for D
- `l::Int`: Which one of the four parameters, l ∈ {1,2,3,4}
- `ev_array::Array{Float64,4}`: eigenvalue array from `generate_eigenval_array`
- `lefti::Index`: left bond index
- `upi::Index`: site index
- `downi::Index`: primed site index

# Returns
- `ITensor`: local MPO tensor for site N
"""
function build_mpo_tensor_end(X::Int, l::Int, ev_array::Array{Float64,4},
    lefti::Index, upi::Index, downi::Index)
  num_plies = size(ev_array, 3)
  num_angles = size(ev_array, 4)
  T = ITensor(lefti, upi, downi)
  T .= 0.
  for a ∈ 1:num_angles
    h = ev_array[X, l, num_plies, a]
    T[lefti=>1, upi=>a, downi=>a] = h^2
    T[lefti=>2, upi=>a, downi=>a] = sqrt(2)*h
    T[lefti=>3, upi=>a, downi=>a] = 1
  end
  return T
end;
#%%



"""
    build_partial_mpo(X::Int, l::Int,
    sites::Vector{<:Index}, ev_array::Array{Float64,4})

Build MPO for lamination parameter X = A,D and l=1,2,3,4.

# Arguments
- `X::Int`: Select A or D parameters, `X=1` for A and `X=2` for D
- `l::Int`: Which one of the four parameters, l ∈ {1,2,3,4}
- `sites::Vector{<:Index}`: `Index` object for the site indices
- `ev_array::Array{Float64,4}`: eigenvalue array from `generate_eigenval_array`

# Returns
- `MPO`: Resulting MPO
"""
function build_partial_mpo(X::Int, l::Int,
    sites::Vector{<:Index}, ev_array::Array{Float64,4})
  num_plies = size(ev_array, 3)
  num_states = size(ev_array, 4)
  links = [Index(3,"Link,l=$n") for n ∈ 1:(num_plies-1)]
  Hmpo = MPO(sites)
  for n in 1:num_plies
    if n > 1
      lefti = links[n-1]
    end
    if n < num_plies
      righti = links[n]
    end
    site = sites[n]
    upi = site
    downi = site'
    if n == 1
      T = build_mpo_tensor_begin(X, l, ev_array, righti, upi, downi)
    elseif n == num_plies
      T = build_mpo_tensor_end(X, l, ev_array, lefti, upi, downi)
    else
      T = build_mpo_tensor_middle(X, l, n, ev_array, lefti, righti, upi, downi)
    end
    Hmpo[n] = T
  end
  return Hmpo
end

#%%

"""
    build_mpo_tensor_constraint_middle(p_list::Matrix{<:Union{Int,Float64}},
    q_list::Matrix{<:Union{Int,Float64}},
    lefti::Index, righti::Index, upi::Index, downi::Index
  )

Build local tensor for MPO of the penalty term, for sites n = 2,...,num_plies-1
For definition of the vectors p and q, see paper section 4.3

# Arguments
- `p_list::Matrix{<:Union{Int,Float64}}`: Matrix where column s represents vector p⁽ˢ⁾
- `q_list::Matrix{<:Union{Int,Float64}}`: Matrix where column s represents vector q⁽ˢ⁾
- `lefti::Index`: left bond index
- `righti::Index`: right bond index
- `upi::Index`: site index
- `downi::Index`: primed site index

# Returns
- `ITensor`: local tensor in penalty MPO
"""
function build_mpo_tensor_constraint_middle(p_list::Matrix{<:Union{Int,Float64}},
    q_list::Matrix{<:Union{Int,Float64}},
    lefti::Index, righti::Index, upi::Index, downi::Index
  )

  @assert size(p_list) == size(q_list)
  num_states = size(p_list,2)
  bond_dim = size(p_list,1) + 2
  T = ITensor(lefti,righti,upi,downi)
  T .= 0.
  for s ∈ 1:num_states
    T[lefti=>1,righti=>1,upi=>s,downi=>s] = 1.
    T[lefti=>bond_dim,righti=>bond_dim,upi=>s,downi=>s] = 1.
    T[lefti=>1,righti=>2:(bond_dim-1),upi=>s,downi=>s] = p_list[:,s]
    T[lefti=>2:(bond_dim-1),righti=>bond_dim,upi=>s,downi=>s] = q_list[:,s]
  end
  return T
end

#%%

#%%

"""
    build_mpo_tensor_constraint_begin(
    p_list::Matrix{<:Union{Int,Float64}},
    righti::Index, upi::Index, downi::Index
  )

Build local tensor for MPO of the penalty term, at site 1
For definition of the vectors p, see paper section 4.3

# Arguments
- `p_list::Matrix{<:Union{Int,Float64}}`: Matrix where column s represents vector p⁽ˢ⁾
- `righti::Index`: right bond index
- `upi::Index`: site index
- `downi::Index`: primed site index

# Returns
- `ITensor`: local tensor in penalty MPO at site 1
"""
function build_mpo_tensor_constraint_begin(
    p_list::Matrix{<:Union{Int,Float64}},
    righti::Index, upi::Index, downi::Index
  )

  num_states = size(p_list,2)
  bond_dim = size(p_list,1) + 2
  T = ITensor(righti,upi,downi)
  T .= 0.
  for s ∈ 1:num_states
    T[righti=>1,upi=>s,downi=>s] = 1.
    T[righti=>2:(bond_dim-1),upi=>s,downi=>s] = p_list[:,s]
  end
  return T
end

#%%

"""
    build_mpo_tensor_constraint_end(
    q_list::Matrix{<:Union{Int,Float64}},
    lefti::Index, upi::Index, downi::Index
  )

Build local tensor for MPO of the penalty term, at sites num_plies.
For definition of the vectors q, see paper section 4.3

# Arguments
- `q_list::Matrix{<:Union{Int,Float64}}`: Matrix where column s represents vector q⁽ˢ⁾
- `lefti::Index`: left bond index
- `upi::Index`: site index
- `downi::Index`: primed site index

# Returns
- `ITensor`: local tensor in penalty MPO at site num_plies
"""
function build_mpo_tensor_constraint_end(
    q_list::Matrix{<:Union{Int,Float64}},
    lefti::Index, upi::Index, downi::Index
  )

  num_states = size(q_list,2)
  bond_dim = size(q_list,1) + 2
  T = ITensor(lefti,upi,downi)
  T .= 0.
  for s ∈ 1:num_states
    T[lefti=>2:(bond_dim-1),upi=>s,downi=>s] = q_list[:,s]
    T[lefti=>bond_dim,upi=>s,downi=>s] = 1.
  end
  return T
end

"""
    build_mpo_disorientation_constraint(sites::Vector{<:Index}, p_list::Matrix{<:Union{Int,Float64}}, q_list::Matrix{<:Union{Int,Float64}})

Build complete MPO for the penalty for the disorientation constraint.
For definition of the vectors p and q, see paper section 4.3

# Arguments:
- `sites::Vector{<:Index}`: `Index` object for the site indices
- `p_list::Matrix{<:Union{Int,Float64}}`: Matrix where column s represents vector p⁽ˢ⁾
- `q_list::Matrix{<:Union{Int,Float64}}`: Matrix where column s represents vector q⁽ˢ⁾

# Returns:
- `MPO`: Penalty MPO for the disorientation constraint
"""
function build_mpo_disorientation_constraint(sites::Vector{<:Index}, p_list::Matrix{<:Union{Int,Float64}}, q_list::Matrix{<:Union{Int,Float64}})
  num_sites = length(sites)
  @assert num_sites > 1
  @assert size(p_list) == size(q_list)
  bond_dim = size(p_list,1)+2
  links = [Index(bond_dim, "Link,l=$n") for n ∈ 1:(num_sites-1)]
  Hmpo = MPO(sites)

  # first site
  Hmpo[1] = build_mpo_tensor_constraint_begin(p_list,links[1],sites[1],sites[1]')
  
  # last site
  Hmpo[end] = build_mpo_tensor_constraint_end(q_list,links[end],sites[end],sites[end]')

  if num_sites == 2
    return Hmpo
  end

  # second site
  Hmpo[2] = build_mpo_tensor_constraint_middle(p_list,q_list,links[1],links[2],sites[2],sites[2]')

  # fill up other sites with copies
  for n ∈ 3:(num_sites-1)
    Hmpo[n] = replaceinds!(copy(Hmpo[2]),[links[1],links[2],sites[2],sites[2]'],[links[n-1],links[n],sites[n],sites[n]'])
  end

  return Hmpo
end

#%%

function angles_diff(a1::Real,a2::Real)
  d = abs(a1 - a2) % 180 # 360
  if d > 90 # 180
    return 180 - d # 360
  end
  return d
end

#%%

"""
    generate_disorientation_constraint_pq_list(
  angles::Vector{<:Union{Int,Float64}},distance::Union{Float64,Int},penalty::Float64
)

Generate vectors p and q for the penalty MPO, according to the paper, section 4.3

# Arguments
- `angles::Vector{<:Union{Int,Float64}}`: possible ply angles
- `distance::Union{Float64,Int}`: angle distance for disorientation constraint
- `penalty::Float64`: penalty for a single constraint violation

# Returns:
- `Matrix{Int}`: Matrix where column s represents vector p⁽ˢ⁾
- `Matrix{Int}`: Matrix where column s represents vector q⁽ˢ⁾
"""
function generate_disorientation_constraint_pq_list(
  angles::Vector{<:Union{Int,Float64}},distance::Union{Float64,Int},penalty::Float64
)
  num_angles = length(angles)
  p_list = Matrix(1.0LinearAlgebra.I, num_angles, num_angles)
  q_list = zeros(Float64,num_angles,num_angles)
  
  for j ∈ 1:num_angles, i ∈ 1:num_angles
    q_list[i,j] = angles_diff(angles[i],angles[j]) > distance ? penalty : 0.
  end

  return p_list,q_list
end

#%%