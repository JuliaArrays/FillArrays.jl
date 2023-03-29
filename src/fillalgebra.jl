## vec

vec(a::Ones{T}) where T = Ones{T}(length(a))
vec(a::Zeros{T}) where T = Zeros{T}(length(a))
vec(a::Fill{T}) where T = Fill{T}(a.value,length(a))

## Transpose/Adjoint
# cannot do this for vectors since that would destroy scalar dot product


transpose(a::OnesMatrix{T}) where T = Ones{T}(reverse(a.axes))
adjoint(a::OnesMatrix{T}) where T = Ones{T}(reverse(a.axes))
transpose(a::ZerosMatrix{T}) where T = Zeros{T}(reverse(a.axes))
adjoint(a::ZerosMatrix{T}) where T = Zeros{T}(reverse(a.axes))
transpose(a::FillMatrix{T}) where T = Fill{T}(transpose(a.value), reverse(a.axes))
adjoint(a::FillMatrix{T}) where T = Fill{T}(adjoint(a.value), reverse(a.axes))

permutedims(a::AbstractFillVector) = fillsimilar(a, (1, length(a)))
permutedims(a::AbstractFillMatrix) = fillsimilar(a, reverse(a.axes))

function permutedims(B::AbstractFill, perm)
    dimsB = size(B)
    ndimsB = length(dimsB)
    (ndimsB == length(perm) && isperm(perm)) || throw(ArgumentError("no valid permutation of dimensions"))
    dimsP = ntuple(i->dimsB[perm[i]], ndimsB)::typeof(dimsB)
    fillsimilar(B, dimsP)
end

Base.@propagate_inbounds function reverse(A::AbstractFill, start::Integer, stop::Integer=lastindex(A))
    @boundscheck checkbounds(A, start)
    @boundscheck checkbounds(A, stop)
    A
end
reverse(A::AbstractFill; dims=:) = A

## Algebraic identities


function mult_fill(a::AbstractFill, b::AbstractFillMatrix)
    axes(a, 2) ≠ axes(b, 1) &&
        throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
    return Fill(getindex_value(a)*getindex_value(b)*size(a,2), (axes(a, 1), axes(b, 2)))
end

function mult_fill(a::AbstractFill, b::AbstractFillVector)
    axes(a, 2) ≠ axes(b, 1) &&
        throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
    return Fill(getindex_value(a)*getindex_value(b)*size(a,2), (axes(a, 1),))
end

function mult_ones(a::AbstractVector, b::AbstractMatrix)
    axes(a, 2) ≠ axes(b, 1) &&
        throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
    return Ones{promote_type(eltype(a), eltype(b))}((axes(a, 1), axes(b, 2)))
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

*(a::AbstractFillVector, b::AbstractFillMatrix) = mult_fill(a,b)
*(a::AbstractFillMatrix, b::AbstractFillMatrix) = mult_fill(a,b)
*(a::AbstractFillMatrix, b::AbstractFillVector) = mult_fill(a,b)

*(a::OnesVector, b::OnesMatrix) = mult_ones(a, b)

*(a::ZerosVector, b::ZerosMatrix) = mult_zeros(a, b)
*(a::ZerosMatrix, b::ZerosMatrix) = mult_zeros(a, b)
*(a::ZerosMatrix, b::ZerosVector) = mult_zeros(a, b)

*(a::ZerosVector, b::AbstractFillMatrix) = mult_zeros(a, b)
*(a::ZerosMatrix, b::AbstractFillMatrix) = mult_zeros(a, b)
*(a::ZerosMatrix, b::AbstractFillVector) = mult_zeros(a, b)
*(a::AbstractFillVector, b::ZerosMatrix) = mult_zeros(a,b)
*(a::AbstractFillMatrix, b::ZerosMatrix) = mult_zeros(a,b)
*(a::AbstractFillMatrix, b::ZerosVector) = mult_zeros(a,b)

*(a::ZerosVector, b::AbstractMatrix) = mult_zeros(a, b)
*(a::ZerosMatrix, b::AbstractMatrix) = mult_zeros(a, b)
*(a::AbstractMatrix, b::ZerosVector) = mult_zeros(a, b)
*(a::AbstractMatrix, b::ZerosMatrix) = mult_zeros(a, b)
*(a::ZerosVector, b::AbstractVector) = mult_zeros(a, b)
*(a::ZerosMatrix, b::AbstractVector) = mult_zeros(a, b)
*(a::AbstractVector, b::ZerosMatrix) = mult_zeros(a, b)

*(a::ZerosVector, b::AdjOrTransAbsVec) = mult_zeros(a, b)

*(a::ZerosVector, b::Diagonal) = mult_zeros(a, b)
*(a::ZerosMatrix, b::Diagonal) = mult_zeros(a, b)
*(a::Diagonal, b::ZerosVector) = mult_zeros(a, b)
*(a::Diagonal, b::ZerosMatrix) = mult_zeros(a, b)
function *(a::Diagonal, b::AbstractFillMatrix)
    size(a,2) == size(b,1) || throw(DimensionMismatch("A has dimensions $(size(a)) but B has dimensions $(size(b))"))
    a.diag .* b # use special broadcast
end
function *(a::AbstractFillMatrix, b::Diagonal)
    size(a,2) == size(b,1) || throw(DimensionMismatch("A has dimensions $(size(a)) but B has dimensions $(size(b))"))
    a .* permutedims(b.diag) # use special broadcast
end

*(a::Adjoint{T, <:StridedMatrix{T}},   b::FillVector{T}) where T = reshape(sum(conj.(parent(a)); dims=1) .* b.value, size(parent(a), 2))
*(a::Transpose{T, <:StridedMatrix{T}}, b::FillVector{T}) where T = reshape(sum(parent(a); dims=1) .* b.value, size(parent(a), 2))
*(a::StridedMatrix{T}, b::FillVector{T}) where T         = reshape(sum(a; dims=2) .* b.value, size(a, 1))

function *(a::Adjoint{T, <:StridedMatrix{T}}, b::FillMatrix{T}) where T
    fB = similar(parent(a), size(b, 1), size(b, 2))
    fill!(fB, b.value)
    return a*fB
end

function *(a::Transpose{T, <:StridedMatrix{T}}, b::FillMatrix{T}) where T
    fB = similar(parent(a), size(b, 1), size(b, 2))
    fill!(fB, b.value)
    return a*fB
end

function *(a::StridedMatrix{T}, b::FillMatrix{T}) where T
    fB = similar(a, size(b, 1), size(b, 2))
    fill!(fB, b.value)
    return a*fB
end
function _adjvec_mul_zeros(a, b)
    la, lb = length(a), length(b)
    if la ≠ lb
        throw(DimensionMismatch("dot product arguments have lengths $la and $lb"))
    end
    return zero(Base.promote_op(*, eltype(a), eltype(b)))
end

*(a::AdjointAbsVec{<:Any,<:ZerosVector}, b::AbstractMatrix) = (b' * a')'
*(a::AdjointAbsVec{<:Any,<:ZerosVector}, b::ZerosMatrix) = (b' * a')'
*(a::TransposeAbsVec{<:Any,<:ZerosVector}, b::AbstractMatrix) = transpose(transpose(b) * transpose(a))
*(a::TransposeAbsVec{<:Any,<:ZerosVector}, b::ZerosMatrix) = transpose(transpose(b) * transpose(a))

*(a::AbstractVector, b::AdjOrTransAbsVec{<:Any,<:ZerosVector}) = a * permutedims(parent(b))
*(a::AbstractMatrix, b::AdjOrTransAbsVec{<:Any,<:ZerosVector}) = a * permutedims(parent(b))
*(a::ZerosVector, b::AdjOrTransAbsVec{<:Any,<:ZerosVector}) = a * permutedims(parent(b))
*(a::ZerosMatrix, b::AdjOrTransAbsVec{<:Any,<:ZerosVector}) = a * permutedims(parent(b))

*(a::AdjointAbsVec, b::ZerosVector) = _adjvec_mul_zeros(a, b)
*(a::AdjointAbsVec{<:Number}, b::ZerosVector{<:Number}) = _adjvec_mul_zeros(a, b)
*(a::TransposeAbsVec, b::ZerosVector) = _adjvec_mul_zeros(a, b)
*(a::TransposeAbsVec{<:Number}, b::ZerosVector{<:Number}) = _adjvec_mul_zeros(a, b)

*(a::Adjoint{T, <:AbstractMatrix{T}} where T, b::Zeros{<:Any, 1}) = mult_zeros(a, b)

function *(a::Transpose{T, <:AbstractVector{T}}, b::ZerosVector{T}) where T<:Real
    la, lb = length(a), length(b)
    if la ≠ lb
        throw(DimensionMismatch("dot product arguments have lengths $la and $lb"))
    end
    return zero(T)
end
*(a::Transpose{T, <:AbstractMatrix{T}}, b::ZerosVector{T}) where T<:Real = mult_zeros(a, b)

# support types with fast sum

function _fill_dot(a::AbstractVector{T}, b::AbstractFill{V}) where {T,V}
    axes(a) == axes(b) || throw(DimensionMismatch("dot product arguments have lengths $(length(a)) and $(length(b))"))
    S = promote_op(+, T, V)
    # treat zero separately to support ∞-vectors
    if iszero(b) || iszero(a)
        zero(S)
    else
        dot(S(sum(a)), S(getindex_value(b)))
    end
end

function _fill_dot_conj(a::AbstractVector{T}, b::AbstractFill{V}) where {T,V}
    axes(a) == axes(b) || throw(DimensionMismatch("dot product arguments have lengths $(length(a)) and $(length(b))"))
    S = promote_op(+, T, V)
    # treat zero separately to support ∞-vectors
    if iszero(b) || iszero(a)
        zero(S)
    else
        dot(S(getindex_value(b)), S(sum(a)))
    end
end

dot(a::AbstractFillVector, b::AbstractFillVector) = _fill_dot(a, b)
dot(a::AbstractFillVector, b::AbstractVector) = _fill_dot_conj(b, a)
dot(a::AbstractVector, b::AbstractFillVector) = _fill_dot(a, b)

function dot(u::AbstractVector, E::Eye, v::AbstractVector)
    length(u) == size(E,1) && length(v) == size(E,2) ||
        throw(DimensionMismatch("dot product arguments have dimensions $(length(u))×$(size(E))×$(length(v))"))
    dot(u, v)
end

function dot(u::AbstractVector, D::Diagonal{<:Any,<:Fill}, v::AbstractVector)
    length(u) == size(D,1) && length(v) == size(D,2) ||
        throw(DimensionMismatch("dot product arguments have dimensions $(length(u))×$(size(D))×$(length(v))"))
    D.diag.value*dot(u, v)
end

function dot(u::AbstractVector{T}, D::Diagonal{U,<:Zeros}, v::AbstractVector{V}) where {T,U,V}
    length(u) == size(D,1) && length(v) == size(D,2) ||
        throw(DimensionMismatch("dot product arguments have dimensions $(length(u))×$(size(D))×$(length(v))"))
    zero(promote_type(T,U,V))
end

# Addition and Subtraction
function +(a::Zeros{T}, b::Zeros{V}) where {T, V} # for disambiguity
    promote_shape(a,b)
    return elconvert(promote_op(+,T,V),a)
end
# no AbstractArray. Otherwise incompatible with StaticArrays.jl
# AbstractFill for disambiguity
for TYPE in (:Array, :AbstractFill, :AbstractRange, :Diagonal)
    @eval function +(a::$TYPE{T}, b::Zeros{V}) where {T, V}
        promote_shape(a,b)
        return elconvert(promote_op(+,T,V),a)
    end
    @eval +(a::Zeros, b::$TYPE) = b + a
end

# for VERSION other than 1.6, could use ZerosMatrix only
function +(a::AbstractFillMatrix{T}, b::UniformScaling) where {T}
    n = checksquare(a)
    return a + Diagonal(Fill(zero(T) + b.λ, n))
end

# LinearAlgebra defines `-(a::AbstractMatrix, b::UniformScaling) = a + (-b)`,
# so the implementation of `-(a::UniformScaling, b::AbstractFill{<:Any,2})` is sufficient
-(a::UniformScaling, b::AbstractFill) = -b + a # @test I-Zeros(3,3) === Diagonal(Ones(3))

-(a::Ones, b::Ones) = Zeros(a) + Zeros(b)

# no AbstractArray. Otherwise incompatible with StaticArrays.jl
for TYPE in (:Array, :AbstractRange)
    @eval begin
        +(a::$TYPE, b::AbstractFill) = fill_add(a, b)
        -(a::$TYPE, b::AbstractFill) = a + (-b)
        +(a::AbstractFill, b::$TYPE) = fill_add(b, a)
        -(a::AbstractFill, b::$TYPE) = a + (-b)
    end
end
+(a::AbstractFill, b::AbstractFill) = Fill(getindex_value(a) + getindex_value(b), promote_shape(a,b))
-(a::AbstractFill, b::AbstractFill) = a + (-b)

@inline function fill_add(a::AbstractArray, b::AbstractFill)
    promote_shape(a, b)
    a .+ [getindex_value(b)]
end
@inline function fill_add(a::AbstractArray{<:Number}, b::AbstractFill)
    promote_shape(a, b)
    a .+ getindex_value(b)
end

# following needed since as of Julia v1.8 convert(AbstractArray{T}, ::AbstractRange) might return a Vector
@inline elconvert(::Type{T}, A::AbstractRange) where T = T(first(A)):T(step(A)):T(last(A))
@inline elconvert(::Type{T}, A::AbstractUnitRange) where T<:Integer = AbstractUnitRange{T}(A)
@inline elconvert(::Type{T}, A::AbstractArray) where T = AbstractArray{T}(A)

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


###
# lmul!/rmul!
###

function lmul!(x::Number, z::AbstractFill)
    λ = getindex_value(z)
    # Following check ensures consistency w/ lmul!(x, Array(z))
    # for, e.g., lmul!(NaN, z)
    x*λ == λ || throw(ArgumentError("Cannot scale by $x"))
    z
end

function rmul!(z::AbstractFill, x::Number)
    λ = getindex_value(z)
    # Following check ensures consistency w/ lmul!(x, Array(z))
    # for, e.g., lmul!(NaN, z)
    λ*x == λ || throw(ArgumentError("Cannot scale by $x"))
    z
end

fillzero(::Type{Fill{T,N,AXIS}}, n, m) where {T,N,AXIS} = Fill{T,N,AXIS}(zero(T), (n, m))
fillzero(::Type{Zeros{T,N,AXIS}}, n, m) where {T,N,AXIS} = Zeros{T,N,AXIS}((n, m))
fillzero(::Type{F}, n, m) where F = throw(ArgumentError("Cannot create a zero array of type $F"))

diagzero(D::Diagonal{F}, i, j) where F<:AbstractFill = fillzero(F, axes(D.diag[i], 1), axes(D.diag[j], 2))
