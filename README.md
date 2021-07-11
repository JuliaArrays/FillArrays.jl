# FillArrays.jl


[![Build Status](https://github.com/JuliaArrays/FillArrays.jl/workflows/CI/badge.svg)](https://github.com/JuliaArrays/FillArrays.jl/actions)
[![codecov](https://codecov.io/gh/JuliaArrays/FillArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaArrays/FillArrays.jl)

Julia package to lazily represent matrices filled with a single entry,
as well as identity matrices.  This package exports the following types:
`Eye`, `Fill`, `Ones`, `Zeros`, `Trues` and `Falses`.


The primary purpose of this package is to present a unified way of constructing
matrices. For example, to construct a 5-by-5 `CLArray` of all zeros, one would use
```julia
julia> CLArray(Zeros(5,5))
```
Because `Zeros` is lazy, this can be accomplished on the GPU with no memory transfer.
Similarly, to construct a 5-by-5 `BandedMatrix` of all zeros with bandwidths `(1,2)`, one would use  
```julia
julia> BandedMatrix(Zeros(5,5), (1, 2))
```

## Usage

Here are the matrix types:
```julia
julia> Zeros(5, 6)
5×6 Zeros{Float64}

julia> Zeros{Int}(2, 3)
2×3 Zeros{Int64}

julia> Ones{Int}(5)
5-element Ones{Int64}

julia> Eye{Int}(5)
 5×5 Diagonal{Int64,Ones{Int64,1,Tuple{Base.OneTo{Int64}}}}:
  1  ⋅  ⋅  ⋅  ⋅
  ⋅  1  ⋅  ⋅  ⋅
  ⋅  ⋅  1  ⋅  ⋅
  ⋅  ⋅  ⋅  1  ⋅
  ⋅  ⋅  ⋅  ⋅  1

julia> Fill(7.0f0, 3, 2)
3×2 Fill{Float32}: entries equal to 7.0

julia> Trues(2, 3)
2×3 Ones{Bool}

julia> Falses(2)
2-element Zeros{Bool}
```

They support conversion to other matrix types like `Array`, `SparseVector`, `SparseMatrix`, and `Diagonal`:
```julia
julia> Matrix(Zeros(5, 5))
5×5 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> SparseMatrixCSC(Zeros(5, 5))
5×5 SparseMatrixCSC{Float64,Int64} with 0 stored entries

julia> Array(Fill(7, (2,3)))
2×3 Array{Int64,2}:
 7  7  7
 7  7  7
```

There is also support for offset index ranges,
and the type includes the `axes`:
```julia
julia> Ones((-3:2, 1:2))
6×2 Ones{Float64,2,Tuple{UnitRange{Int64},UnitRange{Int64}}} with indices -3:2×1:2

julia> Fill(7, ((0:2), (-1:0)))
3×2 Fill{Int64,2,Tuple{UnitRange{Int64},UnitRange{Int64}}} with indices 0:2×-1:0: entries equal to 7

julia> typeof(Zeros(5,6))
Zeros{Float64,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}
```

These types have methods that perform many operations efficiently,
including elementary algebra operations like multiplication and addition,
as well as linear algebra methods like
`norm`, `adjoint`, `transpose` and `vec`.

Broadcasting operations and `map`, `mapreduce` are also done efficiently, when possible, by evaluating the function being applied only once. Notice that this will only match the behaviour of a dense matrix from `fill` if the function is pure:

```julia
julia> sin.(Fill(pi, 10)) isa Fill  # one evaluation, not 10
true

julia> map(sqrt, Fill(4, 2,5))
2×5 Fill{Float64}: entries equal to 2.0

julia> map(_ -> rand(), Fill(4, 2,5))
2×5 Fill{Float64}: entries equal to 0.7201617100284206

julia> map(_ -> rand(), fill(4, 2,5))  # 10 evaluations
2×5 Matrix{Float64}:
 0.43675   0.270809  0.56536   0.0948089  0.24655
 0.959363  0.79598   0.238662  0.401909   0.317716
```
