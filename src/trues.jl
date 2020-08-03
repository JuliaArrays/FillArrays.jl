
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


# y[mask] = x when mask isa Trues
function Base.setindex!(y::AbstractArray{T,N}, x, mask::Trues{N}) where {T,N}
	@boundscheck size(x) == size(mask) == size(y) || throw(DimensionMismatch())
	@boundscheck checkbounds(x, mask)
	@boundscheck axes(mask) == axes(x) || throw("axes mismatch")
    copyto!(y, x)
end

# x[mask] when mask isa Trues
function Base.getindex(x::AbstractArray{T,D}, mask::Trues{D}) where {T,D}
	@boundscheck size(x) == size(mask) || throw(DimensionMismatch())
	@boundscheck checkbounds(x, mask)
	return vec(x)
end
