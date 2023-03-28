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

# OneElement(val, inds::Int...) = OneElement(val, inds)

Base.size(A::OneElement) = map(length, A.axes)
Base.axes(A::OneElement) = A.axes
Base.getindex(A::OneElement{T,N}, i::Vararg{Int,N}) where {T,N} = ifelse(i==A.ind, A.val, zero(T))