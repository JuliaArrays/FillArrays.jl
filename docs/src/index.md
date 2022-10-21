```@meta
DocTestSetup  = quote
    using FillArrays
end
```

# Introduction

`FillArrays` allows one to lazily represent arrays filled with a single entry, as well as identity matrices. This package exports the following types: `Eye`, `Fill`, `Ones`, `Zeros`, `Trues` and `Falses`.

## Quick Start

Create a 2x2 zero matrix

```jldoctest
julia> z = Zeros(2,2)
2×2 Zeros{Float64}

julia> Array(z)
2×2 Matrix{Float64}:
 0.0  0.0
 0.0  0.0
```

We may specify the element type as

```jldoctest
julia> z = Zeros{Int}(2,2)
2×2 Zeros{Int64}

julia> Array(z)
2×2 Matrix{Int64}:
 0  0
 0  0
```

We may create arrays with any number of dimensions. A `Vector` of ones may be created as

```jldoctest
julia> a = Ones(4)
4-element Ones{Float64}

julia> Array(a)
4-element Vector{Float64}:
 1.0
 1.0
 1.0
 1.0
```

Similarly, a `2x3x2` array, where every element is equal to `10`, may be created as

```jldoctest
julia> f = Fill(10, 2,3,2)
2×3×2 Fill{Int64}, with entries equal to 10

julia> Array(f)
2×3×2 Array{Int64, 3}:
[:, :, 1] =
 10  10  10
 10  10  10

[:, :, 2] =
 10  10  10
 10  10  10
```

The elements of a `Fill` array don't need to be restricted to numbers, and these may be any Julia object. For example, we may construct an array of strings using

```jldoctest
julia> f = Fill("hello", 2,5)
2×5 Fill{String}, with entries equal to hello

julia> Array(f)
2×5 Matrix{String}:
 "hello"  "hello"  "hello"  "hello"  "hello"
 "hello"  "hello"  "hello"  "hello"  "hello"
```

### Conversion to a sparse form

These `Fill` array types may be converted to sparse arrays as well, which might be useful in certain cases
```jldoctest sparse
julia> using SparseArrays

julia> z = Zeros{Int}(2,2)
2×2 Zeros{Int64}

julia> sparse(z)
2×2 SparseMatrixCSC{Int64, Int64} with 0 stored entries:
 ⋅  ⋅
 ⋅  ⋅
```
Note, however, that most `Fill` arrays are not sparse, despite being lazily evaluated.

These types have methods that perform many operations efficiently, including elementary algebra operations like multiplication and addition, as well as linear algebra methods like `norm`, `adjoint`, `transpose` and `vec`.

### Custom axes

The various `Fill` equivalents all support offset or custom axes, where instead of the size, one may pass a `Tuple` of axes. So, for example, one may use a `SOneTo` axis from [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl) to construct a statically sized `Fill`.

```jldoctest
julia> using StaticArrays

julia> f = Fill(2, (SOneTo(4), SOneTo(5)))
4×5 Fill{Int64, 2, Tuple{SOneTo{4}, SOneTo{5}}} with indices SOneTo(4)×SOneTo(5), with entries equal to 2
```

The size of such an array would be known at compile time, permitting compiler optimizations.

We may construct infinite fill arrays by passing infinite-sized axes, see [`InfiniteArrays.jl`](https://github.com/JuliaArrays/InfiniteArrays.jl).

### Other lazy types

A lazy representation of an identity matrix may be constructured using `Eye`. For example, a `4x4` identity matrix with `Float32` elements may be constructed as

```jldoctest sparse
julia> id = Eye{Float32}(4)
4×4 Eye{Float32}

julia> Array(id)
4×4 Matrix{Float32}:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0
 0.0  0.0  0.0  1.0

julia> sparse(id)
4×4 SparseMatrixCSC{Float32, Int64} with 4 stored entries:
 1.0   ⋅    ⋅    ⋅
  ⋅   1.0   ⋅    ⋅
  ⋅    ⋅   1.0   ⋅
  ⋅    ⋅    ⋅   1.0
```

Note that an `Eye` actually returns a `Diagonal` matrix, where the diagonal is a `Ones` vector.

## Warning about map and broadcasting

Broadcasting operations, and `map` and `mapreduce`, are also done efficiently, by evaluating the function being applied only once:

```jldoctest
julia> map(sqrt, Fill(4, 2,5))  # one evaluation, not 10, to save time
2×5 Fill{Float64}, with entries equal to 2.0

julia> println.(Fill(pi, 10))
π
10-element Fill{Nothing}, with entries equal to nothing
```

Notice that this will only match the behaviour of a dense matrix from `fill` if the function is pure. And that this shortcut is taken before any other fused broadcast:

```julia
julia> map(_ -> rand(), Fill("pi", 2,5))  # not a pure function!
2×5 Fill{Float64, with entries equal to 0.7201617100284206

julia> map(_ -> rand(), fill("4", 2,5))  # 10 evaluations, different answer!
2×5 Matrix{Float64}:
 0.43675   0.270809  0.56536   0.0948089  0.24655
 0.959363  0.79598   0.238662  0.401909   0.317716

julia> ones(1,5) .+ (_ -> rand()).(Fill("vec", 2))  # Fill broadcast is done first
2×5 Matrix{Float64}:
 1.51796  1.51796  1.51796  1.51796  1.51796
 1.51796  1.51796  1.51796  1.51796  1.51796

julia> ones(1,5) .+ (_ -> rand()).(fill("vec", 2))  # fused, 10 evaluations
2×5 Matrix{Float64}:
 1.51337  1.17578  1.19815  1.43035  1.2987
 1.30253  1.21909  1.61755  1.02645  1.77681
```

# API

```@autodocs
Modules = [FillArrays]
```
