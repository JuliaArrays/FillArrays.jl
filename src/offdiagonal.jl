"""
    OffDiagonal(band::AbstractVector, k::Integer, axes)
    OffDiagonal(band::AbstractVector, k::Integer, m::Integer, n::Integer)
    OffDiagonal(band::AbstractVector, k::Integer)

A matrix whose only non-zero entries lie on its `k`-th diagonal, where they are given by `band`.
As for `diag`, `k > 0` refers to a superdiagonal and `k < 0` to a subdiagonal, and
`length(band)` must equal the length of that diagonal. If the size is omitted, the matrix is
square with `length(band) + abs(k)` rows.

Indexing a `Diagonal` of an `AbstractFill`, a `RectDiagonal` or an `OffDiagonal` by unit ranges
returns an `OffDiagonal`.

# Examples

```jldoctest
julia> FillArrays.OffDiagonal([1, 2, 3], 1)
4×4 FillArrays.OffDiagonal{Int64, Vector{Int64}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}:
 ⋅  1  ⋅  ⋅
 ⋅  ⋅  2  ⋅
 ⋅  ⋅  ⋅  3
 ⋅  ⋅  ⋅  ⋅

julia> A = Eye(5)[3:5, 1:4];

julia> A.k
2

julia> A.band
2-element Ones{Float64}
```
"""
struct OffDiagonal{T,V<:AbstractVector{T},Axes<:Tuple{Vararg{AbstractUnitRange,2}}} <: AbstractMatrix{T}
    band::V
    k::Int
    axes::Axes

    @inline function OffDiagonal{T,V,Axes}(band::V, k::Integer, axes::Axes) where {T,V<:AbstractVector{T},Axes<:Tuple{Vararg{AbstractUnitRange,2}}}
        Base.require_one_based_indexing(band)
        len = _bandlength(map(length, axes)..., k)
        length(band) == len || throw(DimensionMismatch(LazyString("band has length ", length(band), " but diagonal ", k, " of a ", Base.dims2string(map(length, axes)), " matrix has length ", len)))
        A = new{T,V,Axes}(band, k, axes)
        Base.require_one_based_indexing(A)
        return A
    end
end

# number of entries on the k-th diagonal of an m × n matrix, which is zero when the diagonal lies outside the matrix
_bandlength(m, n, k) = max(0, k ≥ 0 ? min(m, n - k) : min(m + k, n))

@inline OffDiagonal{T,V}(band::V, k::Integer, axes::Axes) where {T,V,Axes<:Tuple{Vararg{AbstractUnitRange,2}}} = OffDiagonal{T,V,Axes}(band, k, axes)
@inline OffDiagonal{T,V}(band::V, k::Integer, sz::Tuple{Vararg{Integer,2}}) where {T,V} = OffDiagonal{T,V}(band, k, oneto.(sz))
@inline OffDiagonal{T,V}(band::V, k::Integer, m::Integer, n::Integer) where {T,V} = OffDiagonal{T,V}(band, k, (m, n))
@inline function OffDiagonal{T,V}(band::V, k::Integer) where {T,V}
    n = length(band) + abs(k)
    OffDiagonal{T,V}(band, k, (n, n))
end
@inline function OffDiagonal{T}(band::AbstractVector, args...) where T
    b = convert(AbstractVector{T}, band)
    OffDiagonal{T,typeof(b)}(b, args...)
end
@inline OffDiagonal(band::V, args...) where {V<:AbstractVector} = OffDiagonal{eltype(V),V}(band, args...)

axes(A::OffDiagonal) = A.axes
size(A::OffDiagonal) = map(length, A.axes)

@inline function getindex(A::OffDiagonal{T}, i::Integer, j::Integer) where T
    @boundscheck checkbounds(A, i, j)
    if j - i == A.k
        @inbounds r = A.band[min(i, j)]
    else
        r = zero(T)
    end
    return r
end

function setindex!(A::OffDiagonal, v, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    if j - i == A.k
        @inbounds A.band[min(i, j)] = v
    elseif !iszero(v)
        throw(ArgumentError(LazyString("cannot set entry (", i, ", ", j, ") off diagonal ", A.k, " to a nonzero value (", v, ")")))
    end
    return v
end

function diag(A::OffDiagonal{T}, k::Integer=0) where T
    k == A.k && return copy(A.band)
    fill!(similar(A.band, length(diagind(A, k))), zero(T))
end

Base.replace_in_print_matrix(A::OffDiagonal, i::Integer, j::Integer, s::AbstractString) =
    j - i == A.k ? s : Base.replace_with_centered_mark(s)

##
# Slicing a matrix supported on a single diagonal by unit ranges keeps a single diagonal
##

const SingleBandMatrix = Union{Diagonal{<:Number,<:AbstractFillVector}, RectDiagonal, OffDiagonal}

_band(D::Union{Diagonal,RectDiagonal}) = D.diag
_band(A::OffDiagonal) = A.band
_bandindex(::Union{Diagonal,RectDiagonal}) = 0
_bandindex(A::OffDiagonal) = A.k

Base.@propagate_inbounds function getindex(A::SingleBandMatrix, kr::Union{AbstractUnitRange{<:Integer},Colon}, jr::Union{AbstractUnitRange{<:Integer},Colon})
    I = (_onebased_index(A, 1, kr), _onebased_index(A, 2, jr))
    _singleband_getindex(Base.index_shape(I...), A, I...)
end

# all SingleBandMatrix are one-based, so a Colon may be replaced by a OneTo even if the axes are not OneTo
_onebased_index(A, d, ::Colon) = oneto(size(A, d))
_onebased_index(_, _, r) = r

# indices with offset or infinite axes use the generic fallback
Base.@propagate_inbounds _singleband_getindex(_, A, kr, jr) = invoke(getindex, Tuple{AbstractArray, Vararg{Any}}, A, kr, jr)

@inline function _singleband_getindex(shape::Tuple{Base.OneTo,Base.OneTo}, A, kr, jr)
    @boundscheck checkbounds(A, kr, jr)
    # entry (i, j) of the result is A[kr[i], jr[j]], which lies on the band when jr[j] - kr[i] == _bandindex(A)
    k = _bandindex(A) + first(kr) - first(jr)
    m, n = length(kr), length(jr)
    len = _bandlength(m, n, k)
    # the band of A is indexed by min(row, col), so start from the first entry of the k-th diagonal of the result
    st = min(first(kr) + max(0, -k), first(jr) + max(0, k))
    @inbounds band = _band(A)[st:st+len-1]
    OffDiagonal(band, k, shape)
end
