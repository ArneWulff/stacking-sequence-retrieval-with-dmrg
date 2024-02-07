"""
    lamination_parameters(angles)

Calculate A and D lamination_parameters
for a symmetric stack with an even number of plies.

angles: angles of individual plies (in degrees)

"""
function lamination_parameters(angles::Union{Vector{Float64},Vector{Int}})
  num_plies = length(angles)
  lps = zeros(8)

  # A parameters
  lps[1] = sum(cospi.(2 * angles / 180)) / num_plies
  lps[2] = sum(sinpi.(2 * angles / 180)) / num_plies
  lps[3] = sum(cospi.(4 * angles / 180)) / num_plies
  lps[4] = sum(sinpi.(4 * angles / 180)) / num_plies

  # D parameters
  boundaries_3 = (range(0, num_plies) / num_plies) .^ 3
  weights = boundaries_3[2:end] - boundaries_3[1:end-1]
  lps[5] = sum(weights .* cospi.(2 * angles / 180))
  lps[6] = sum(weights .* sinpi.(2 * angles / 180))
  lps[7] = sum(weights .* cospi.(4 * angles / 180))
  lps[8] = sum(weights .* sinpi.(4 * angles / 180))

  return lps
end