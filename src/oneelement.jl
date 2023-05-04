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

function _mulonel!(y, A, x::OneElement{<:Any,1}, alpha::Number, beta::Number)
    check_matmul_sizes(y, A, x)
    if x.ind[1] ∉ axes(x,1) # in this case x is all zeros
        mul!(y, A, Zeros{eltype(x)}(axes(x)), alpha, beta)
        return y
    end
    αx = alpha * x.val
    if iszero(beta)
        y .= αx .* view(A, :, x.ind[1])
    else
        y .= αx .* view(A, :, x.ind[1]) .+ beta .* y
    end
    y
end

function _mulonel!(C, A, B::OneElement{<:Any,2}, alpha::Number, beta::Number)
    check_matmul_sizes(C, A, B)
    αB = alpha * B.val
    if B.ind[1] ∉ axes(B,1) || B.ind[2] ∉ axes(B,2) # in this case x is all zeros
        mul!(C, A, Zeros{eltype(B)}(axes(B)), alpha, beta)
        return C
    end
    if iszero(beta)
        C .= zero(eltype(C))
        C[:, B.ind[2]] .= αB .* view(A, :, B.ind[1])
    else
        lmul!(beta, C)
        C[:, B.ind[2]] .+= αB .* view(A, :, B.ind[1])
    end
    C
end

for MT in (:StridedMatrix, :(Transpose{<:Any, <:StridedMatrix}), :(Adjoint{<:Any, <:StridedMatrix}))
    @eval function mul!(y::StridedVector, A::$MT, x::OneElement{<:Any,1}, alpha::Number, beta::Number)
        _mulonel!(y, A, x, alpha, beta)
    end
    @eval function mul!(y::StridedMatrix, A::$MT, x::OneElement{<:Any,2}, alpha::Number, beta::Number)
        _mulonel!(y, A, x, alpha, beta)
    end
end
