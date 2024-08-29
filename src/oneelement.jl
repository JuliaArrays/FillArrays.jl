"""
    OneElement(val, ind, axesorsize) <: AbstractArray

Represents an array with the specified axes (if its a tuple of `AbstractUnitRange`s)
or size (if its a tuple of `Integer`s), with a single entry set to `val` and all others equal to zero,
specified by `ind``.
"""
struct OneElement{T,N,I,A} <: AbstractArray{T,N}
  val::T
  ind::I
  axes::A
  OneElement(val::T, ind::I, axes::A) where {T, I<:NTuple{N,Int}, A<:NTuple{N,AbstractUnitRange}} where {N} = new{T,N,I,A}(val, ind, axes)
  OneElement(val::T, ind::Tuple{}, axes::Tuple{}) where {T} = new{T,0,Tuple{},Tuple{}}(val, ind, axes)
end

const OneElementVector{T,I,A} = OneElement{T,1,I,A}
const OneElementMatrix{T,I,A} = OneElement{T,2,I,A}
const OneElementVecOrMat{T,I,A} = Union{OneElementVector{T,I,A}, OneElementMatrix{T,I,A}}

OneElement(val, inds::NTuple{N,Int}, sz::NTuple{N,Integer}) where N = OneElement(val, inds, oneto.(sz))
"""
    OneElement(val, ind::Int, n::Int)

Creates a length `n` vector where the `ind` entry is equal to `val`, and all other entries are zero.
"""
OneElement(val, ind::Int, len::Int) = OneElement(val, (ind,), (len,))
"""
    OneElement(ind::Int, n::Int)

Creates a length `n` vector where the `ind` entry is equal to `1`, and all other entries are zero.
"""
OneElement(inds::Int, sz::Int) = OneElement(1, inds, sz)
OneElement{T}(val, inds::NTuple{N,Int}, sz::NTuple{N,Integer}) where {T,N} = OneElement(convert(T,val), inds, oneto.(sz))
OneElement{T}(val, inds::Int, sz::Int) where T = OneElement{T}(val, (inds,), (sz,))

"""
    OneElement{T}(ind::Int, n::Int)

Creates a length `n` vector where the `ind` entry is equal to `one(T)`, and all other entries are zero.
"""
OneElement{T}(inds::Int, sz::Int) where T = OneElement(one(T), inds, sz)

Base.size(A::OneElement) = map(length, A.axes)
Base.axes(A::OneElement) = A.axes
Base.getindex(A::OneElement{T,0}) where {T} = getindex_value(A)
Base.@propagate_inbounds function Base.getindex(A::OneElement{T,N}, kj::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, kj...)
    ifelse(kj == A.ind, A.val, zero(T))
end
const VectorInds = Union{AbstractUnitRange{<:Integer}, Integer} # no index is repeated for these indices
const VectorIndsWithColon = Union{VectorInds, Colon}
# retain the values from Ainds corresponding to the vector indices in inds
_index_shape(Ainds, inds::Tuple{Integer, Vararg{Any}}) = _index_shape(Base.tail(Ainds), Base.tail(inds))
_index_shape(Ainds, inds::Tuple{AbstractVector, Vararg{Any}}) = (Ainds[1], _index_shape(Base.tail(Ainds), Base.tail(inds))...)
_index_shape(::Tuple{}, ::Tuple{}) = ()
Base.@propagate_inbounds function Base.getindex(A::OneElement{T,N}, inds::Vararg{VectorIndsWithColon,N}) where {T,N}
    I = to_indices(A, inds) # handle Bool, and convert to compatible index types
    @boundscheck checkbounds(A, I...)
    shape = _index_shape(I, I)
    nzind = _index_shape(A.ind, I) .- first.(shape) .+ firstindex.(shape)
    containsval = all(in.(A.ind, I))
    OneElement(getindex_value(A), containsval ? Int.(nzind) : Int.(lastindex.(shape,1)).+1, axes.(shape,1))
end

"""
    nzind(A::OneElement{T,N}) -> CartesianIndex{N}

Return the index where `A` contains a non-zero value.

!!! note
    The indices are not guaranteed to lie within the valid index bounds for `A`,
    and if `FillArrays.nzind(A) ∉ CartesianIndices(A)` then `all(iszero, A)`.
    On the other hand, if `FillArrays.nzind(A) in CartesianIndices(A)` then
    `A[FillArrays.nzind(A)] == FillArrays.getindex_value(A)`

# Examples
```jldoctest
julia> A = OneElement(2, (1,2), (2,2))
2×2 OneElement{Int64, 2, Tuple{Int64, Int64}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}:
 ⋅  2
 ⋅  ⋅

julia> FillArrays.nzind(A)
CartesianIndex(1, 2)

julia> A[FillArrays.nzind(A)]
2
```
"""
nzind(f::OneElement) = CartesianIndex(f.ind)

"""
    getindex_value(A::OneElement)

Return the only non-zero value stored in `A`.

!!! note
    If the index at which the value is stored doesn't lie within the valid indices of `A`, then
    this returns `zero(eltype(A))`.

# Examples
```jldoctest
julia> A = OneElement(2, 3)
3-element OneElement{Int64, 1, Tuple{Int64}, Tuple{Base.OneTo{Int64}}}:
 ⋅
 1
 ⋅

julia> FillArrays.getindex_value(A)
1
```
"""
getindex_value(A::OneElement) = all(in.(A.ind, axes(A))) ? A.val : zero(eltype(A))

@inline function Base.isassigned(F::OneElement, i::Integer...)
    @boundscheck checkbounds(Bool, F, to_indices(F, i)...) || return false
    return true
end

Base.AbstractArray{T,N}(A::OneElement{<:Any,N}) where {T,N} = OneElement(T(A.val), A.ind, A.axes)

Base.replace_in_print_matrix(o::OneElementVector, k::Integer, j::Integer, s::AbstractString) =
    o.ind == (k,) ? s : Base.replace_with_centered_mark(s)

Base.replace_in_print_matrix(o::OneElementMatrix, k::Integer, j::Integer, s::AbstractString) =
    o.ind == (k,j) ? s : Base.replace_with_centered_mark(s)

Base.@propagate_inbounds function Base.setindex(A::AbstractZeros{T,N}, v, kj::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, kj...)
    OneElement(convert(T, v), kj, axes(A))
end

zero(A::OneElement) = OneElement(zero(A.val), A.ind, A.axes)

iszero(A::OneElement) = iszero(getindex_value(A))

function isone(A::OneElementMatrix)
    lenA = length(A)
    lenA == 0 && return true
    lenA > 1 && return false
    isone(getindex_value(A))
end

-(O::OneElement) = OneElement(-O.val, O.ind, O.axes)

*(x::OneElement, b::Number) = OneElement(x.val * b, x.ind, x.axes)
*(b::Number, x::OneElement) = OneElement(b * x.val, x.ind, x.axes)
/(x::OneElement, b::Number) = OneElement(x.val / b, x.ind, x.axes)
\(b::Number, x::OneElement) = OneElement(b \ x.val, x.ind, x.axes)

# matrix-vector and matrix-matrix multiplication

# Fill and OneElement
function *(A::OneElementMatrix, B::OneElementVecOrMat)
    check_matmul_sizes(A, B)
    valA = getindex_value(A)
    valB = getindex_value(B)
    val = valA * valB * (A.ind[2] == B.ind[1])
    OneElement(val, (A.ind[1], B.ind[2:end]...), (axes(A,1), axes(B)[2:end]...))
end

*(A::OneElementMatrix, x::AbstractZerosVector) = mult_zeros(A, x)

function *(A::OneElementMatrix, B::AbstractFillVector)
    check_matmul_sizes(A, B)
    val = getindex_value(A) * getindex_value(B)
    OneElement(val, A.ind[1], size(A,1))
end

# Special matrix types

function *(A::OneElementMatrix, D::Diagonal)
    check_matmul_sizes(A, D)
    nzcol = A.ind[2]
    val = if nzcol in axes(D,1)
        A.val * D[nzcol, nzcol]
    else
        A.val * zero(eltype(D))
    end
    OneElement(val, A.ind, size(A))
end
function *(D::Diagonal, A::OneElementMatrix)
    check_matmul_sizes(D, A)
    nzrow = A.ind[1]
    val = if nzrow in axes(D,2)
        D[nzrow, nzrow] * A.val
    else
        zero(eltype(D)) * A.val
    end
    OneElement(val, A.ind, size(A))
end

# Inplace multiplication

# We use this for out overloads for _mul! for OneElement because its more efficient
# due to how efficient 2 arg mul is when one or more of the args are OneElement
function __mulonel!(C, A, B, alpha, beta)
    ABα = A * B * alpha
    if iszero(beta)
        C .= ABα
    else
        C .= ABα .+ C .* beta
    end
    return C
end
# These methods remove the ambituity in _mul!. This isn't strictly necessary, but this makes Aqua happy.
function _mul!(C::AbstractVector, A::OneElementMatrix, B::OneElementVector, alpha, beta)
    __mulonel!(C, A, B, alpha, beta)
end
function _mul!(C::AbstractMatrix, A::OneElementMatrix, B::OneElementMatrix, alpha, beta)
    __mulonel!(C, A, B, alpha, beta)
end

function mul!(C::AbstractMatrix, A::OneElementMatrix, B::OneElementMatrix, alpha::Number, beta::Number)
    _mul!(C, A, B, alpha, beta)
end
function mul!(C::AbstractVector, A::OneElementMatrix, B::OneElementVector, alpha::Number, beta::Number)
    _mul!(C, A, B, alpha, beta)
end

@inline function __mul!(y, A::AbstractMatrix, x::OneElement, alpha, beta)
    xα = Ref(x.val * alpha)
    ind1 = x.ind[1]
    if iszero(beta)
        y .= view(A, :, ind1) .* xα
    else
        y .= view(A, :, ind1) .* xα .+ y .* beta
    end
    return y
end

function _mul!(y::AbstractVector, A::AbstractMatrix, x::OneElementVector, alpha, beta)
    check_matmul_sizes(y, A, x)
    if iszero(getindex_value(x))
        mul!(y, A, Zeros{eltype(x)}(axes(x)), alpha, beta)
        return y
    end
    __mul!(y, A, x, alpha, beta)
    y
end

function _mul!(C::AbstractMatrix, A::AbstractMatrix, B::OneElementMatrix, alpha, beta)
    check_matmul_sizes(C, A, B)
    if iszero(getindex_value(B))
        mul!(C, A, Zeros{eltype(B)}(axes(B)), alpha, beta)
        return C
    end
    nzrow, nzcol = B.ind
    if iszero(beta)
        C .= Ref(zero(eltype(C)))
    else
        view(C, :, 1:nzcol-1) .*= beta
        view(C, :, nzcol+1:size(C,2)) .*= beta
    end
    y = view(C, :, nzcol)
    __mul!(y, A, B, alpha, beta)
    C
end
function _mul!(C::AbstractMatrix, A::Diagonal, B::OneElementMatrix, alpha, beta)
    check_matmul_sizes(C, A, B)
    if iszero(getindex_value(B))
        mul!(C, A, Zeros{eltype(B)}(axes(B)), alpha, beta)
        return C
    end
    nzrow, nzcol = B.ind
    ABα = A * B * alpha
    if iszero(beta)
        C .= Ref(zero(eltype(C)))
        C[nzrow, nzcol] = ABα[nzrow, nzcol]
    else
        view(C, :, 1:nzcol-1) .*= beta
        view(C, :, nzcol+1:size(C,2)) .*= beta
        y = view(C, :, nzcol)
        y .= view(ABα, :, nzcol) .+ y .* beta
    end
    C
end

function _mul!(C::AbstractMatrix, A::OneElementMatrix, B::AbstractMatrix, alpha, beta)
    check_matmul_sizes(C, A, B)
    if iszero(getindex_value(A))
        mul!(C, Zeros{eltype(A)}(axes(A)), B, alpha, beta)
        return C
    end
    nzrow, nzcol = A.ind
    y = view(C, nzrow, :)
    Aval = A.val
    if iszero(beta)
        C .= Ref(zero(eltype(C)))
        y .= Ref(Aval) .* view(B, nzcol, :) .* alpha
    else
        view(C, 1:nzrow-1, :) .*= beta
        view(C, nzrow+1:size(C,1), :) .*= beta
        y .= Ref(Aval) .* view(B, nzcol, :) .* alpha .+ y .* beta
    end
    C
end
function _mul!(C::AbstractMatrix, A::OneElementMatrix, B::Diagonal, alpha, beta)
    check_matmul_sizes(C, A, B)
    if iszero(getindex_value(A))
        mul!(C, Zeros{eltype(A)}(axes(A)), B, alpha, beta)
        return C
    end
    nzrow, nzcol = A.ind
    ABα = A * B * alpha
    if iszero(beta)
        C .= Ref(zero(eltype(C)))
        C[nzrow, nzcol] = ABα[nzrow, nzcol]
    else
        view(C, 1:nzrow-1, :) .*= beta
        view(C, nzrow+1:size(C,1), :) .*= beta
        y = view(C, nzrow, :)
        y .= view(ABα, nzrow, :) .+ y .* beta
    end
    C
end

function _mul!(C::AbstractVector, A::OneElementMatrix, B::AbstractVector, alpha, beta)
    check_matmul_sizes(C, A, B)
    if iszero(getindex_value(A))
        mul!(C, Zeros{eltype(A)}(axes(A)), B, alpha, beta)
        return C
    end
    nzrow, nzcol = A.ind
    Aval = A.val
    if iszero(beta)
        C .= Ref(zero(eltype(C)))
        C[nzrow] = Aval * B[nzcol] * alpha
    else
        view(C, 1:nzrow-1) .*= beta
        view(C, nzrow+1:size(C,1)) .*= beta
        C[nzrow] = Aval * B[nzcol] * alpha + C[nzrow] * beta
    end
    C
end

for MT in (:StridedMatrix, :(Transpose{<:Any, <:StridedMatrix}), :(Adjoint{<:Any, <:StridedMatrix}))
    @eval function mul!(y::StridedVector, A::$MT, x::OneElementVector, alpha::Number, beta::Number)
        _mul!(y, A, x, alpha, beta)
    end
end
for MT in (:StridedMatrix, :(Transpose{<:Any, <:StridedMatrix}), :(Adjoint{<:Any, <:StridedMatrix}),
            :Diagonal)
    @eval function mul!(C::StridedMatrix, A::$MT, B::OneElementMatrix, alpha::Number, beta::Number)
        _mul!(C, A, B, alpha, beta)
    end
    @eval function mul!(C::StridedMatrix, A::OneElementMatrix, B::$MT, alpha::Number, beta::Number)
        _mul!(C, A, B, alpha, beta)
    end
end
function mul!(C::StridedVector, A::OneElementMatrix, B::StridedVector, alpha::Number, beta::Number)
    _mul!(C, A, B, alpha, beta)
end

function mul!(y::AbstractVector, A::AbstractFillMatrix, x::OneElementVector, alpha::Number, beta::Number)
    _mul!(y, A, x, alpha, beta)
end
function mul!(C::AbstractMatrix, A::AbstractFillMatrix, B::OneElementMatrix, alpha::Number, beta::Number)
    _mul!(C, A, B, alpha, beta)
end
function mul!(C::AbstractVector, A::OneElementMatrix, B::AbstractFillVector, alpha::Number, beta::Number)
    _mul!(C, A, B, alpha, beta)
end
function mul!(C::AbstractMatrix, A::OneElementMatrix, B::AbstractFillMatrix, alpha::Number, beta::Number)
    _mul!(C, A, B, alpha, beta)
end

# adjoint/transpose

adjoint(A::OneElementMatrix) = OneElement(adjoint(A.val), reverse(A.ind), reverse(A.axes))
transpose(A::OneElementMatrix) = OneElement(transpose(A.val), reverse(A.ind), reverse(A.axes))

# isbanded
function LinearAlgebra.isbanded(A::OneElementMatrix, kl::Integer, ku::Integer)
    iszero(getindex_value(A)) || kl <= A.ind[2] - A.ind[1] <= ku
end

# tril/triu

function tril(A::OneElementMatrix, k::Integer=0)
    nzband = A.ind[2] - A.ind[1]
    OneElement(nzband > k ? zero(A.val) : A.val, A.ind, axes(A))
end

function triu(A::OneElementMatrix, k::Integer=0)
    nzband = A.ind[2] - A.ind[1]
    OneElement(nzband < k ? zero(A.val) : A.val, A.ind, axes(A))
end


# issymmetric
issymmetric(O::OneElement) = axes(O,1) == axes(O,2) && isdiag(O) && issymmetric(getindex_value(O))
ishermitian(O::OneElement) = axes(O,1) == axes(O,2) && isdiag(O) && ishermitian(getindex_value(O))

# diag
function diag(O::OneElementMatrix, k::Integer=0)
    Base.require_one_based_indexing(O)
    len = length(diagind(O, k))
    ind = O.ind[2] - O.ind[1] == k ? (k >= 0 ? O.ind[2] - k : O.ind[1] + k) : len + 1
    OneElement(getindex_value(O), ind, len)
end

# broadcast

for f in (:abs, :abs2, :conj, :real, :imag)
    @eval function broadcasted(::DefaultArrayStyle{N}, ::typeof($f), r::OneElement{<:Any,N}) where {N}
        OneElement($f(r.val), r.ind, axes(r))
    end
end
function broadcasted(::DefaultArrayStyle{N}, ::typeof(^), r::OneElement{<:Any,N}, x::Number) where {N}
    OneElement(r.val^x, r.ind, axes(r))
end
function broadcasted(::DefaultArrayStyle{N}, ::typeof(*), r::OneElement{<:Any,N}, x::Number) where {N}
    OneElement(r.val*x, r.ind, axes(r))
end
function broadcasted(::DefaultArrayStyle{N}, ::typeof(/), r::OneElement{<:Any,N}, x::Number) where {N}
    OneElement(r.val/x, r.ind, axes(r))
end
function broadcasted(::DefaultArrayStyle{N}, ::typeof(\), x::Number, r::OneElement{<:Any,N}) where {N}
    OneElement(x \ r.val, r.ind, axes(r))
end

# reshape

function Base.reshape(A::OneElement, shape::Tuple{Vararg{Int}})
    prod(shape) == length(A) || throw(DimensionMismatch("new dimension $shape must be consistent with array size $(length(A))"))
    if all(in.(A.ind, axes(A)))
        # we use the fact that the linear index of the non-zero value is preserved
        oldlinind = LinearIndices(A)[A.ind...]
        newcartind = CartesianIndices(shape)[oldlinind]
    else
        # arbitrarily set to some value outside the domain
        newcartind = shape .+ 1
    end
    OneElement(A.val, Tuple(newcartind), shape)
end

#permute
_permute(x, p) = ntuple(i -> x[p[i]], length(x))
permutedims(o::OneElementMatrix) = OneElement(o.val, reverse(o.ind), reverse(o.axes))
permutedims(o::OneElementVector) = reshape(o, (1, length(o)))
permutedims(o::OneElement, dims) = OneElement(o.val, _permute(o.ind, dims), _permute(o.axes, dims))

# unique
function unique(O::OneElement)
    v = getindex_value(O)
    len = iszero(v) ? 1 : min(2, length(O))
    OneElement(getindex_value(O), len, len)
end
allunique(O::OneElement) = length(O) <= 1 || (length(O) < 3 && !iszero(getindex_value(O)))

# show
_maybesize(t::Tuple{Base.OneTo{Int}, Vararg{Base.OneTo{Int}}}) = size.(t,1)
_maybesize(t) = t
Base.show(io::IO, A::OneElement) = print(io, OneElement, "(", A.val, ", ", A.ind, ", ", _maybesize(axes(A)), ")")
Base.show(io::IO, A::OneElement{<:Any,1,Tuple{Int},Tuple{Base.OneTo{Int}}}) =
    print(io, OneElement, "(", A.val, ", ", A.ind[1], ", ", size(A,1), ")")

# mapreduce
Base.sum(O::OneElement; dims=:, kw...) = _sum(O, dims; kw...)
_sum(O::OneElement, ::Colon; kw...) = sum((getindex_value(O),); kw...)
function _sum(O::OneElement, dims; kw...)
    v = _sum(O, :; kw...)
    ax = Base.reduced_indices(axes(O), dims)
    ind = ntuple(x -> x in dims ? first(ax[x]) + (O.ind[x] in axes(O)[x]) - 1 : O.ind[x], ndims(O))
    OneElement(v, ind, ax)
end
