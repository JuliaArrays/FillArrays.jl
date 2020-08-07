
"""
    Trues = Ones{Bool, N, Axes} where {N, Axes}

Lazy version of `trues` with axes.
Typically created using `Trues(dims)` or `Trues(dims...)`

# Example
```jldoctest
julia> Trues(1,3)
1×3 Ones{Bool,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}} = true

julia> Trues((2,3))
2×3 Ones{Bool,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}} = true
```
"""
const Trues = Ones{Bool, N, Axes} where {N, Axes}


""" `Falses = Zeros{Bool, N, Axes}` (see `Trues`) """
const Falses = Zeros{Bool, N, Axes} where {N, Axes}


# y[mask] = x when mask isa Trues (cf y[:] = x)
# Supported here only for arrays with standard OneTo axes.
function Base.setindex!(y::AbstractArray{T,N}, x,
    mask::Trues{N, NTuple{N,Base.OneTo{Int}}},
) where {T,N}
    if axes(x) isa NTuple{N,Base.OneTo{Int}} &&
       axes(y) isa NTuple{N,Base.OneTo{Int}}
        @boundscheck size(y) == size(mask) || throw(BoundsError(y, mask))
        @boundscheck size(x) == size(mask) || throw(DimensionMismatch(
            "tried to assign $(length(x)) elements to $(length(y)) destinations"))
        @boundscheck checkbounds(y, mask)
        return copyto!(y, x)
    end
    return setindex!(y, x, trues(size(mask))) # fall back on usual setindex!
end

# x[mask] when mask isa Trues (cf x[trues(size(x))] or x[:])
# Supported here only for arrays with standard OneTo axes.
function Base.getindex(x::AbstractArray{T,N},
    mask::Trues{N, NTuple{N,Base.OneTo{Int}}},
) where {T,N}
    if axes(x) isa NTuple{N,Base.OneTo{Int}} where N
       @boundscheck size(x) == size(mask) || throw(BoundsError(x, mask))
       return vec(x)
    end
    return x[trues(size(x))] # else revert to usual getindex method
end
