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
  OneElement(val::T, ind::I, axes::A) where {T<:Number, I<:NTuple{N,Int}, A<:NTuple{N,AbstractUnitRange}} where {N} = new{T,N,I,A}(val, ind, axes)
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
    OneElement{T}(val, ind::Int, n::Int)

Creates a length `n` vector where the `ind` entry is equal to `one(T)`, and all other entries are zero.
"""
OneElement{T}(inds::Int, sz::Int) where T = OneElement(one(T), inds, sz)

Base.size(A::OneElement) = map(length, A.axes)
Base.axes(A::OneElement) = A.axes
function Base.getindex(A::OneElement{T,N}, kj::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, kj...)
    ifelse(kj == A.ind, A.val, zero(T))
end

Base.AbstractArray{T,N}(A::OneElement{<:Any,N}) where {T,N} = OneElement(T(A.val), A.ind, A.axes)

Base.replace_in_print_matrix(o::OneElementVector, k::Integer, j::Integer, s::AbstractString) =
    o.ind == (k,) ? s : Base.replace_with_centered_mark(s)

Base.replace_in_print_matrix(o::OneElementMatrix, k::Integer, j::Integer, s::AbstractString) =
    o.ind == (k,j) ? s : Base.replace_with_centered_mark(s)

function Base.setindex(A::Zeros{T,N}, v, kj::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, kj...)
    OneElement(convert(T, v), kj, axes(A))
end

# multiplication
# Fill and OneElement

function *(A::AbstractFillMatrix, x::OneElementVector)
    check_matmul_sizes(A, x)
    val = getindex_value(A) * (x.ind[1] in axes(x,1) ? x.val : zero(eltype(x)))
    Fill(val, (axes(A,1),))
end

function *(A::OneElementMatrix, B::AbstractFillVector)
    check_matmul_sizes(A, B)
    val = (A.ind[2] in axes(A,2) ? A.val : zero(eltype(A))) * getindex_value(B)
    OneElement(val, A.ind[1], size(A,1))
end

@inline function __mulonel!(y, A, x, alpha, beta)
    αx = alpha * x.val
    ind1 = x.ind[1]
    if iszero(beta)
        y .= αx .* view(A, :, ind1)
    else
        y .= αx .* view(A, :, ind1) .+ beta .* y
    end
    return y
end

function _mulonel!(y, A, x::OneElementVector, alpha::Number, beta::Number)
    check_matmul_sizes(y, A, x)
    if x.ind[1] ∉ axes(x,1) # in this case x is all zeros
        mul!(y, A, Zeros{eltype(x)}(axes(x)), alpha, beta)
        return y
    end
    __mulonel!(y, A, x, alpha, beta)
    y
end

function _mulonel!(C, A, B::OneElementMatrix, alpha::Number, beta::Number)
    check_matmul_sizes(C, A, B)
    if B.ind[1] ∉ axes(B,1) || B.ind[2] ∉ axes(B,2) # in this case x is all zeros
        mul!(C, A, Zeros{eltype(B)}(axes(B)), alpha, beta)
        return C
    end
    y = @view C[:, B.ind[2]]
    __mulonel!(y, A, B, alpha, beta)
    C
end

for MT in (:StridedMatrix, :(Transpose{<:Any, <:StridedMatrix}), :(Adjoint{<:Any, <:StridedMatrix}))
    @eval function mul!(y::StridedVector, A::$MT, x::OneElementVector, alpha::Number, beta::Number)
        _mulonel!(y, A, x, alpha, beta)
    end
    @eval function mul!(C::StridedMatrix, A::$MT, B::OneElementMatrix, alpha::Number, beta::Number)
        _mulonel!(C, A, B, alpha, beta)
    end
end

function mul!(y::AbstractVector, A::AbstractFillMatrix, x::OneElementVector, alpha::Number, beta::Number)
    _mulonel!(y, A, x, alpha, beta)
end
function mul!(C::AbstractMatrix, A::AbstractFillMatrix, B::OneElementMatrix, alpha::Number, beta::Number)
    _mulonel!(C, A, B, alpha, beta)
end
