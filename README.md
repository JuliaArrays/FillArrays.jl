# FillArrays.jl

[![Build Status](https://travis-ci.org/JuliaMatrices/BandedMatrices.jl.svg?branch=master)](https://travis-ci.org/JuliaMatrices/BandedMatrices.jl)

Julia package to lazily representing matrices filled with a single entry,
as well as identity matrices.  This package exports the following types: `Eye`,
`Fill`, `Ones`, and `Zeros`.


The primary purpose of this package is to precent a unified way of constructing
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
julia> Zeros(5, 5)
5×5 FillArrays.Zeros{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> Zeros{Int}(5, 5)
5×5 FillArrays.Zeros{Int64,2}:
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0

julia> Ones{Int}(5)
5-element FillArrays.Ones{Int64,1}:
 1
 1
 1
 1
 1

julia> Eye{Int}(5,6)
5×6 FillArrays.Eye{Int64}:
 1  0  0  0  0  0
 0  1  0  0  0  0
 0  0  1  0  0  0
 0  0  0  1  0  0
 0  0  0  0  1  0

julia> Fill(5.0f0, 3)
3-element FillArrays.Fill{Float32,1}:
 5.0
 5.0
 5.0
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
julia> Matrix(Zeros(5, 5))
5×5 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
``` 
