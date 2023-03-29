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

Base.replace_in_print_matrix(o::OneElement{<:Any,2}, k::Integer, j::Integer, s::AbstractString) =
    o.ind == (k,j) ? s : Base.replace_with_centered_mark(s)

function Base.setindex(A::Zeros{T,N}, v, kj::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, kj...)
    OneElement(convert(T, v), kj, axes(A))
end