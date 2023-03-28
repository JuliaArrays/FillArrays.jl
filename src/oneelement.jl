"""
    OneElement(val, ind, axes) <: AbstractArray
Extremely simple `struct` used for the gradient of scalar `getindex`.
"""
struct OneElement{T,N,I,A} <: AbstractArray{T,N}
  val::T
  ind::I
  axes::A
  OneElement(val::T, ind::I, axes::A) where {T<:Number, I<:NTuple{N,Int}, A<:NTuple{N,AbstractUnitRange}} where {N} = new{T,N,I,A}(val, ind, axes)
end

OneElement(val, inds::NTuple{N,Int}, sz::NTuple{N,Integer}) where N = OneElement(val, inds, oneto.(sz))
OneElement(val, inds::Int, sz::Int) = OneElement(val, (inds,), (sz,))
OneElement(inds::Int, sz::Int) = OneElement(1, inds, sz)
OneElement{T}(inds::Int, sz::Int) where T = OneElement(one(T), inds, sz)

Base.size(A::OneElement) = map(length, A.axes)
Base.axes(A::OneElement) = A.axes
Base.getindex(A::OneElement{T,N}, i::Vararg{Int,N}) where {T,N} = ifelse(i==A.ind, A.val, zero(T))

Base.replace_in_print_matrix(o::OneElement{<:Any,2}, k::Integer, j::Integer, s::AbstractString) =
    o.ind == (k,j) ? s : Base.replace_with_centered_mark(s)

function Base.setindex(A::Zeros{T,N}, v, kj::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, kj...)
    OneElement(convert(T, v), kj, axes(A))
end