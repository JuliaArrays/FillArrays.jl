## vec

vec(a::Ones{T}) where T = Ones{T}(length(a))
vec(a::Zeros{T}) where T = Zeros{T}(length(a))
vec(a::Fill{T}) where T = Fill{T}(a.value,length(a))

## Transpose/Adjoint
# cannot do this for vectors since that would destroy scalar dot product


transpose(a::Ones{T,2}) where T = Ones{T}(reverse(a.axes))
adjoint(a::Ones{T,2}) where T = Ones{T}(reverse(a.axes))
transpose(a::Zeros{T,2}) where T = Zeros{T}(reverse(a.axes))
adjoint(a::Zeros{T,2}) where T = Zeros{T}(reverse(a.axes))
transpose(a::Fill{T,2}) where T = Fill{T}(transpose(a.value), reverse(a.axes))
adjoint(a::Fill{T,2}) where T = Fill{T}(adjoint(a.value), reverse(a.axes))

fillsimilar(a::Ones{T}, axes) where T = Ones{T}(axes)
fillsimilar(a::Zeros{T}, axes) where T = Zeros{T}(axes)
fillsimilar(a::AbstractFill, axes) = Fill(getindex_value(a), axes)

permutedims(a::AbstractFill{<:Any,1}) = fillsimilar(a, (1, length(a)))
permutedims(a::AbstractFill{<:Any,2}) = fillsimilar(a, reverse(a.axes))

function permutedims(B::AbstractFill, perm)
    dimsB = size(B)
    ndimsB = length(dimsB)
    (ndimsB == length(perm) && isperm(perm)) || throw(ArgumentError("no valid permutation of dimensions"))
    dimsP = ntuple(i->dimsB[perm[i]], ndimsB)::typeof(dimsB)
    fillsimilar(B, dimsP)
end

## Algebraic identities


function mult_fill(a::AbstractFill, b::AbstractFill{<:Any,2})
    axes(a, 2) ≠ axes(b, 1) &&
        throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
    return Fill(getindex_value(a)*getindex_value(b), (axes(a, 1), axes(b, 2)))
end

function mult_fill(a::AbstractFill, b::AbstractFill{<:Any,1})
    axes(a, 2) ≠ axes(b, 1) &&
        throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
    return Fill(getindex_value(a)*getindex_value(b), (axes(a, 1),))
end

function mult_ones(a, b::AbstractMatrix)
    axes(a, 2) ≠ axes(b, 1) &&
        throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
    return Ones{promote_type(eltype(a), eltype(b))}((axes(a, 1), axes(b, 2)))
end
function mult_ones(a, b::AbstractVector)
    axes(a, 2) ≠ axes(b, 1) &&
        throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
    return Ones{promote_type(eltype(a), eltype(b))}((axes(a, 1),))
end

function mult_zeros(a, b::AbstractMatrix)
    axes(a, 2) ≠ axes(b, 1) &&
        throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
    return Zeros{promote_type(eltype(a), eltype(b))}((axes(a, 1), axes(b, 2)))
end
function mult_zeros(a, b::AbstractVector)
    axes(a, 2) ≠ axes(b, 1) &&
        throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
    return Zeros{promote_type(eltype(a), eltype(b))}((axes(a, 1),))
end

*(a::AbstractFill{<:Any,1}, b::AbstractFill{<:Any,2}) = mult_fill(a,b)
*(a::AbstractFill{<:Any,2}, b::AbstractFill{<:Any,2}) = mult_fill(a,b)
*(a::AbstractFill{<:Any,2}, b::AbstractFill{<:Any,1}) = mult_fill(a,b)

*(a::Ones{<:Any,1}, b::Ones{<:Any,2}) = mult_ones(a, b)
*(a::Ones{<:Any,2}, b::Ones{<:Any,2}) = mult_ones(a, b)
*(a::Ones{<:Any,2}, b::Ones{<:Any,1}) = mult_ones(a, b)

*(a::Zeros{<:Any,1}, b::Zeros{<:Any,2}) = mult_zeros(a, b)
*(a::Zeros{<:Any,2}, b::Zeros{<:Any,2}) = mult_zeros(a, b)
*(a::Zeros{<:Any,2}, b::Zeros{<:Any,1}) = mult_zeros(a, b)

*(a::Zeros{<:Any,1}, b::AbstractFill{<:Any,2}) = mult_zeros(a, b)
*(a::Zeros{<:Any,2}, b::AbstractFill{<:Any,2}) = mult_zeros(a, b)
*(a::Zeros{<:Any,2}, b::AbstractFill{<:Any,1}) = mult_zeros(a, b)
*(a::AbstractFill{<:Any,1}, b::Zeros{<:Any,2}) = mult_zeros(a,b)
*(a::AbstractFill{<:Any,2}, b::Zeros{<:Any,2}) = mult_zeros(a,b)
*(a::AbstractFill{<:Any,2}, b::Zeros{<:Any,1}) = mult_zeros(a,b)

*(a::Zeros{<:Any,1}, b::AbstractMatrix) = mult_zeros(a, b)
*(a::Zeros{<:Any,2}, b::AbstractMatrix) = mult_zeros(a, b)
*(a::AbstractMatrix, b::Zeros{<:Any,1}) = mult_zeros(a, b)
*(a::AbstractMatrix, b::Zeros{<:Any,2}) = mult_zeros(a, b)
*(a::Zeros{<:Any,1}, b::AbstractVector) = mult_zeros(a, b)
*(a::Zeros{<:Any,2}, b::AbstractVector) = mult_zeros(a, b)
*(a::AbstractVector, b::Zeros{<:Any,2}) = mult_zeros(a, b)

*(a::Zeros{<:Any,1}, b::Diagonal) = mult_zeros(a, b)
*(a::Zeros{<:Any,2}, b::Diagonal) = mult_zeros(a, b)
*(a::Diagonal, b::Zeros{<:Any,1}) = mult_zeros(a, b)
*(a::Diagonal, b::Zeros{<:Any,2}) = mult_zeros(a, b)


function *(a::Adjoint{T, <:AbstractVector{T}}, b::Zeros{S, 1}) where {T, S}
    la, lb = length(a), length(b)
    if la ≠ lb
        throw(DimensionMismatch("dot product arguments have lengths $la and $lb"))
    end
    return zero(promote_type(T, S))
end
*(a::Adjoint{T, <:AbstractMatrix{T}} where T, b::Zeros{<:Any, 1}) = mult_zeros(a, b)

function *(a::Transpose{T, <:AbstractVector{T}}, b::Zeros{T, 1}) where T<:Real
    la, lb = length(a), length(b)
    if la ≠ lb
        throw(DimensionMismatch("dot product arguments have lengths $la and $lb"))
    end
    return zero(T)
end
*(a::Transpose{T, <:AbstractMatrix{T}}, b::Zeros{T, 1}) where T<:Real = mult_zeros(a, b)


+(a::Zeros) = a
-(a::Zeros) = a

# Zeros +/- Zeros
function +(a::Zeros{T}, b::Zeros{V}) where {T, V}
    size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
    return Zeros{promote_type(T,V)}(size(a)...)
end
-(a::Zeros, b::Zeros) = -(a + b)

# Zeros +/- Fill and Fill +/- Zeros
function +(a::AbstractFill{T}, b::Zeros{V}) where {T, V}
    size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
    return convert(AbstractFill{promote_type(T, V)}, a)
end
+(a::Zeros, b::AbstractFill) = b + a
-(a::AbstractFill, b::Zeros) = a + b
-(a::Zeros, b::AbstractFill) = a + (-b)

# Zeros +/- Array and Array +/- Zeros
function +(a::Zeros{T, N}, b::AbstractArray{V, N}) where {T, V, N}
    size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
    return AbstractArray{promote_type(T,V),N}(b)
end
function +(a::Array{T, N}, b::Zeros{V, N}) where {T, V, N}
    size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
    return AbstractArray{promote_type(T,V),N}(a)
end

function -(a::Zeros{T, N}, b::AbstractArray{V, N}) where {T, V, N}
    size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
    return -b + a
end
-(a::Array{T, N}, b::Zeros{V, N}) where {T, V, N} = a + b


+(a::AbstractRange, b::Zeros) = b + a

function +(a::Zeros{T, 1}, b::AbstractRange) where {T}
    size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
    Tout = promote_type(T, eltype(b))
    return convert(Tout, first(b)):convert(Tout, step(b)):convert(Tout, last(b))
end
function +(a::Zeros{T, 1}, b::UnitRange) where {T}
    size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
    Tout = promote_type(T, eltype(b))
    return convert(Tout, first(b)):convert(Tout, last(b))
end

function -(a::Zeros{T, 1}, b::AbstractRange{V}) where {T, V}
    size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
    return -b + a
end
-(a::AbstractRange{T}, b::Zeros{V, 1}) where {T, V} = a + b



####
# norm
####

for op in (:norm1, :norm2, :normInf, :normMinusInf)
    @eval $op(a::Zeros) = norm(getindex_value(a))
end

normp(a::Zeros, p) = norm(getindex_value(a))

norm1(a::AbstractFill) = length(a)*norm(getindex_value(a))
norm2(a::AbstractFill) = sqrt(length(a))*norm(getindex_value(a))
normp(a::AbstractFill, p) = (length(a))^(1/p)*norm(getindex_value(a))
normInf(a::AbstractFill) = norm(getindex_value(a))
normMinusInf(a::AbstractFill) = norm(getindex_value(a))
