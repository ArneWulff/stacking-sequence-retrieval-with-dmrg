"""
Functions to generate random stacking sequences.

Here, we implemented the functions that are used to generate the
random stacking sequences. Just randomly choosing the ply angles
individually for each ply would predominantly produce stacking
sequences with the ply angles appearing in similar ratios (with
A lamination parameters all close to 0) and where the ply angles
are quite evenly mixed over the stack (with D lamination parameters
all close to the A lamination parameters). To enforce a more 
diverse set of stacking sequences, we used a more elaborate method
to generate random stacking sequences. In a second step (see 
`generate_random_lp.jl`), we use a kernel density estimator to get 
a distribution in lamination parameter space, that is closer to a
uniform distribution, and thereby mitigate biases from our
generation method. 
Futhermore, we already enforce the disorientation constraint in the
random stacking sequence to obtain target lamination parameters,
for which a exact solution exist even when including the
constraint.

For this purpose, the whole stack is divided into segments, of which
each gets randomly assigned probabilites for the individual ply angles 
(functions `generate_probs_stack` and generate_probs).
The function `generate_stack_from_probs` start at one and of the stack
and selects one ply angle at a time according to the probabilties and
considering the disorientation constraint.

"""

using StatsBase: sample, Weights


"""
    generate_probs(num_angles::Int)

Generate probabilities for num_angles angles.
  
# Arguments
- `num_angles::Int`: Number of angles

# Returns
- `Vector{Float64}`: Probabilities as a vector with num_angle elements
"""
function generate_probs(num_angles::Int)
  # sample from gaussian for random direction
  probs = randn(Float64, num_angles)
  while all(==(0.),probs)
    probs = randn(Float64, num_angles)
  end
  
  # square amplitudes
  probs = probs.^2

  # normalize
  probs = probs./sum(probs)
  
  return probs
end


"""
    generate_probs_stack(num_angles::Int,num_segments::Int=5)

Generate probabilities for a whole stack.

The stack is devided in equally sized segments. Each segment gets
assigned its own random probabilities.

# Arguments
- `num_angles::Int`: number of angles

# Keywords
- `num_segments::Int = 5`: number of segments

# Returns
- `Matrix{Float64}`: Matrix of shape (num_angles,num_segments) containing
    the probabilities
"""
function generate_probs_stack(num_angles::Int,num_segments::Int=5)
  probs = Matrix{Float64}(undef,num_angles,num_segments)
  for s ∈ 1:num_segments
    probs[:,s] = generate_probs(num_angles)[:]
  end
  return probs
end

#%%


"""
    angles_diff(a1::Real,a2::Real)

Calculate difference in angles in degrees
"""
function angles_diff(a1::Real,a2::Real)
  d = abs(a1 - a2) % 180 # 360
  if d > 90 # 180
    return 180 - d # 360
  end
  return d
end


"""
    disorientation_constraint_violations(
  angles::Vector{<:Union{Int,Float64}},distance::Union{Float64,Int}
)

Generate a matrix summerizing the constraint violations.

The matrix element at position (i,j) is true, exactly if angles i and j violate a disorientation 
constriant with angle `distance`.

# Arguments
- `angles::Vector{<:Union{Int,Float64}}`: The possible ply angles in degrees
- `distance::Union{Float64,Int}`: The angle distant for the disorientation constraint
    in degrees

# Returns
- `BitMatrix`: A matrix of shape (length(angles),length(angles)) with true for constraint
     violations
"""
function disorientation_constraint_violations(
  angles::Vector{<:Union{Int,Float64}},distance::Union{Float64,Int}
)
  num_angles = length(angles)
  violations = BitMatrix(undef,num_angles,num_angles)
  for j ∈ 1:num_angles, i ∈ 1:num_angles
    violations[i,j] = angles_diff(angles[i],angles[j]) > distance ? 1 : 0
  end
  return violations
end


"""
    softmax(x)

Softmax of x
"""
softmax(x) = begin
    expx = exp.(x)
    expx / sum(expx)
end

#%% 

"""
    generate_stack_from_probs(num_plies::Int,probs::Matrix{Float64},c_list::BitMatrix;reverse=false)

Generate stacking sequence from probabilies that satisfies the 
disorientation constraint.

The function works as follows: It starts at one end of the stack. 
Then each ply-angle is chosen according to the probabilities 

# Arguments
- `num_plies::Int`: number of plies
- `probs::Matrix{Float64}`: probabilites, as generated with `generate_probs_stack`
- `c_list::BitMatrix`: matrix with constraint violations, as generated 
    with `disorientation_constraint_violations`

# Keywords
- `reverse::Bool`: start generating the ply angles in reverse from N to 1

# Returns
- `Vector{Int}`: generated stacking sequence
"""
function generate_stack_from_probs(num_plies::Int,probs::Matrix{Float64},c_list::BitMatrix;reverse=false)
  num_angles,num_segments = size(probs)
  if reverse
    probs = probs[:,end:-1:1]
  end
  
  # get boundaries and segment sizes
  last_of_segment = round.(Int,Vector(1:(num_segments)).*(num_plies/num_segments))
  segment_sizes = [last_of_segment[1],(last_of_segment[2:end]-last_of_segment[1:(end-1)])...]
  
  stack = Vector{Int}(undef,num_plies)

  elements = Vector(1:num_angles)
  
  # first ply (since no constraints)
  stack[1] = sample(elements,Weights(softmax(probs[:,1])))

  for s ∈ 1:num_segments
    # since first ply is already placed in segment 1,
    # got to ply 2
    p1 = s == 1 ? 2 : 1

    # get last ply of last segment
    last_segment_last = s == 1 ? 0 : last_of_segment[s-1]

    # generate ply angles for rest of segment
    for ply ∈ p1:segment_sizes[s]
      n = last_segment_last+ply # number in stack
      angle_last_ply = stack[n-1]
      
      # exlude constraint violating ply-angles
      possible_angles = (!).(c_list[:,angle_last_ply])

      # get selection probabilities without constraint violating ply angles
      selection_probs = probs[:,s][possible_angles]
      selection_probs /= sum(selection_probs)
      
      # randomly choose ply angle according to weights
      stack[n] = sample(elements[possible_angles],Weights(selection_probs))
    end
  end
  
  if reverse
    return stack[end:-1:1]
  end

  return stack
end

#%%