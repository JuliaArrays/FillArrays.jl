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
    OneElement{T}(val, ind::Int, n::Int)

Creates a length `n` vector where the `ind` entry is equal to `one(T)`, and all other entries are zero.
"""
OneElement{T}(inds::Int, sz::Int) where T = OneElement(one(T), inds, sz)

Base.size(A::OneElement) = map(length, A.axes)
Base.axes(A::OneElement) = A.axes
Base.@propagate_inbounds function Base.getindex(A::OneElement{T,N}, kj::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, kj...)
    ifelse(kj == A.ind, A.val, zero(T))
end

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

function *(A::AbstractFillMatrix, x::OneElementVector)
    check_matmul_sizes(A, x)
    val = getindex_value(A) * getindex_value(x)
    Fill(val, (axes(A,1),))
end
*(A::AbstractZerosMatrix, x::OneElementVector) = mult_zeros(A, x)

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
    αx = alpha * x.val
    ind1 = x.ind[1]
    if iszero(beta)
        y .= αx .* view(A, :, ind1)
    else
        y .= αx .* view(A, :, ind1) .+ beta .* y
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
    if iszero(beta)
        C .= zero(eltype(C))
    else
        view(C, :, 1:B.ind[2]-1) .*= beta
        view(C, :, B.ind[2]+1:size(C,2)) .*= beta
    end
    y = view(C, :, B.ind[2])
    __mul!(y, A, B, alpha, beta)
    C
end
function _mul!(C::AbstractMatrix, A::Diagonal, B::OneElementMatrix, alpha, beta)
    check_matmul_sizes(C, A, B)
    if iszero(getindex_value(B))
        mul!(C, A, Zeros{eltype(B)}(axes(B)), alpha, beta)
        return C
    end
    if iszero(beta)
        C .= zero(eltype(C))
    else
        view(C, :, 1:B.ind[2]-1) .*= beta
        view(C, :, B.ind[2]+1:size(C,2)) .*= beta
    end
    ABα = A * B * alpha
    nzrow, nzcol = B.ind
    if iszero(beta)
        C[B.ind...] = ABα[B.ind...]
    else
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
    if iszero(beta)
        C .= zero(eltype(C))
    else
        view(C, 1:A.ind[1]-1, :) .*= beta
        view(C, A.ind[1]+1:size(C,1), :) .*= beta
    end
    y = view(C, A.ind[1], :)
    ind2 = A.ind[2]
    Aval = A.val
    if iszero(beta)
        y .= Aval .* view(B, ind2, :) .* alpha
    else
        y .= Aval .* view(B, ind2, :) .* alpha .+ y .* beta
    end
    C
end
function _mul!(C::AbstractMatrix, A::OneElementMatrix, B::Diagonal, alpha, beta)
    check_matmul_sizes(C, A, B)
    if iszero(getindex_value(A))
        mul!(C, Zeros{eltype(A)}(axes(A)), B, alpha, beta)
        return C
    end
    if iszero(beta)
        C .= zero(eltype(C))
    else
        view(C, 1:A.ind[1]-1, :) .*= beta
        view(C, A.ind[1]+1:size(C,1), :) .*= beta
    end
    ABα = A * B * alpha
    nzrow, nzcol = A.ind
    if iszero(beta)
        C[A.ind...] = ABα[A.ind...]
    else
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
    if iszero(beta)
        C .= zero(eltype(C))
    else
        view(C, 1:nzrow-1) .*= beta
        view(C, nzrow+1:size(C,1)) .*= beta
    end
    Aval = A.val
    if iszero(beta)
        C[nzrow] = Aval * B[nzcol] * alpha
    else
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

# broadcast
function broadcasted(::DefaultArrayStyle{N}, ::typeof(conj), r::OneElement{<:Any,N}) where {N}
    OneElement(conj(r.val), r.ind, axes(r))
end
function broadcasted(::DefaultArrayStyle{N}, ::typeof(real), r::OneElement{<:Any,N}) where {N}
    OneElement(real(r.val), r.ind, axes(r))
end
function broadcasted(::DefaultArrayStyle{N}, ::typeof(imag), r::OneElement{<:Any,N}) where {N}
    OneElement(imag(r.val), r.ind, axes(r))
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

# show
_maybesize(t::Tuple{Base.OneTo{Int}, Vararg{Base.OneTo{Int}}}) = size.(t,1)
_maybesize(t) = t
Base.show(io::IO, A::OneElement) = print(io, OneElement, "(", A.val, ", ", A.ind, ", ", _maybesize(axes(A)), ")")
Base.show(io::IO, A::OneElement{<:Any,1,Tuple{Int},Tuple{Base.OneTo{Int}}}) =
    print(io, OneElement, "(", A.val, ", ", A.ind[1], ", ", size(A,1), ")")
