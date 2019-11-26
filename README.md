# FillArrays.jl

[![Build Status](https://travis-ci.org/JuliaArrays/FillArrays.jl.svg?branch=master)](https://travis-ci.org/JuliaArrays/FillArrays.jl)
[![codecov](https://codecov.io/gh/JuliaArrays/FillArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaArrays/FillArrays.jl)

Julia package to lazily representing matrices filled with a single entry,
as well as identity matrices.  This package exports the following types: `Eye`,
`Fill`, `Ones`, and `Zeros`.


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

Here are the matrix type4s:
```julia
julia> Zeros(5, 6)
5×6 Zeros{Float64,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}:
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

 julia> Zeros{Int}(5, 6)
 5×6 Zeros{Int64,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}:
  0  0  0  0  0  0
  0  0  0  0  0  0
  0  0  0  0  0  0
  0  0  0  0  0  0
  0  0  0  0  0  0

julia> Ones{Int}(5)
5-element Ones{Int64,1,Tuple{Base.OneTo{Int64}}}:
 1
 1
 1
 1
 1

 julia> Eye{Int}(5)
 5×5 Diagonal{Int64,Ones{Int64,1,Tuple{Base.OneTo{Int64}}}}:
  1  ⋅  ⋅  ⋅  ⋅
  ⋅  1  ⋅  ⋅  ⋅
  ⋅  ⋅  1  ⋅  ⋅
  ⋅  ⋅  ⋅  1  ⋅
  ⋅  ⋅  ⋅  ⋅  1

julia> Fill(5.0f0, 3, 2)
3×2 Fill{Float32,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}:
 5.0  5.0
 5.0  5.0
 5.0  5.0
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
```

There is also support for offset index ranges:
```julia
julia> Ones((-3:2, 1:2))
Ones{Float64,2,Tuple{UnitRange{Int64},UnitRange{Int64}}} with indices -3:2×1:2:
 1.0  1.0
 1.0  1.0
 1.0  1.0
 1.0  1.0
 1.0  1.0
 1.0  1.0
```
