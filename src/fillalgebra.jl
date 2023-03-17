## vec

vec(a::Ones{T}) where T = Ones{T}(length(a))
vec(a::Zeros{T}) where T = Zeros{T}(length(a))
vec(a::Fill{T}) where T = Fill{T}(a.value,length(a))

## Transpose/Adjoint
# cannot do this for vectors since that would destroy scalar dot product

for fun in (:transpose,:adjoint)
    for TYPE in (:Ones,:Zeros)
        @eval $fun(a::$TYPE{T,2}) where T = $TYPE{T}(reverse(a.axes))
    end
    @eval $fun(a::FillMatrix{T}) where T = Fill{T}($fun(a.value), reverse(a.axes))
end

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
@inline checkdimensionmismatch(a::AbstractVecOrMat, b::AbstractVecOrMat) = axes(a, 2) ≠ axes(b, 1) && throw(DimensionMismatch("A has axes $(axes(a)) but B has axes $(axes(b))"))
@inline productaxes(a::AbstractVecOrMat, b::AbstractVector) = (axes(a, 1),)
@inline productaxes(a::AbstractVecOrMat, b::AbstractMatrix) = (axes(a, 1), axes(b, 2))

function mult_fill(a::AbstractFill, b::AbstractFillVecOrMat)
    checkdimensionmismatch(a,b)
    return Fill(getindex_value(a)*getindex_value(b)*size(a,2), productaxes(a,b))
end

function mult_ones(a::AbstractVector, b::AbstractMatrix)
    checkdimensionmismatch(a,b)
    return Ones{promote_type(eltype(a), eltype(b))}(productaxes(a,b))
end

function mult_zeros(a, b::AbstractVecOrMat)
    checkdimensionmismatch(a,b)
    return Zeros{promote_type(eltype(a), eltype(b))}(productaxes(a,b))
end

*(a::ZerosVector, b::AdjOrTransAbsVec) = mult_zeros(a, b)

# Matrix * VecOrMat. 
# For Vector*Matrix, LinearAlgebra reshapes the vector to matrix automatically. 
# See *(a::AbstractVector, B::AbstractMatrix) at matmul.jl
*(a::AbstractFillMatrix, b::AbstractFillVecOrMat) = mult_fill(a,b)
*(a::ZerosMatrix, b::ZerosVector) = mult_zeros(a, b)
*(a::ZerosMatrix, b::ZerosMatrix) = mult_zeros(a, b)
*(a::OnesVector, b::OnesMatrix) = mult_ones(a, b)
for TYPE in (AbstractFillMatrix, AbstractMatrix, Diagonal)
    @eval begin
        *(a::ZerosMatrix, b::$TYPE) = mult_zeros(a,b)
        *(a::$TYPE, b::ZerosVector) = mult_zeros(a,b)
        *(a::$TYPE, b::ZerosMatrix) = mult_zeros(a,b)
    end
end
for TYPE in (:AbstractFillVector, :AbstractVector)
    @eval *(a::ZerosMatrix, b::$TYPE) = mult_zeros(a,b)
end

function *(a::Diagonal, b::AbstractFill{<:Any,2})
    checkdimensionmismatch(a,b)
    a.diag .* b # use special broadcast
end
function *(a::AbstractFill{<:Any,2}, b::Diagonal)
    checkdimensionmismatch(a,b)
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

# AdjOrTrans{ZerosVector} * Matrix
*(a::AdjointAbsVec{<:Any,<:ZerosVector}, b::AbstractMatrix) = (b' * a')'
*(a::AdjointAbsVec{<:Any,<:ZerosVector}, b::ZerosMatrix) = (b' * a')'
*(a::TransposeAbsVec{<:Any,<:ZerosVector}, b::AbstractMatrix) = transpose(transpose(b) * transpose(a))
*(a::TransposeAbsVec{<:Any,<:ZerosVector}, b::ZerosMatrix) = transpose(transpose(b) * transpose(a))

# VecOrMat * AdjOrTrans{ZerosVector}
for TYPE in (:AbstractVector, :AbstractMatrix, :ZerosVector, :ZerosMatrix)
    @eval *(a::$TYPE, b::AdjOrTransAbsVec{<:Any,<:ZerosVector}) = a * permutedims(parent(b))
end

# AdjOrTrans{Vector} * ZerosVector
for T1 in (:AdjointAbsVec, :TransposeAbsVec), T2 in (:Any, :Number)
    @eval *(a::$T1{<:$T2}, b::ZerosVector{<:$T2}) = _adjvec_mul_zeros(a, b)
end

*(a::Adjoint{T, <:AbstractMatrix{T}} where T, b::ZerosVector) = mult_zeros(a, b)

function *(a::Transpose{T, <:AbstractVector{T}}, b::ZerosVector{T}) where T<:Real
    la, lb = length(a), length(b)
    if la ≠ lb
        throw(DimensionMismatch("dot product arguments have lengths $la and $lb"))
    end
    return zero(T)
end
*(a::Transpose{T, <:AbstractMatrix{T}}, b::ZerosVector{T}) where T<:Real = mult_zeros(a, b)

# treat zero separately to support ∞-vectors
function _fill_dot(a::AbstractVector, b::AbstractVector)
    axes(a) == axes(b) || throw(DimensionMismatch("dot product arguments have lengths $(length(a)) and $(length(b))"))
    if iszero(a) || iszero(b)
        zero(promote_type(eltype(a),eltype(b)))
    elseif isa(a,AbstractFill)
        getindex_value(a)sum(b)
    else
        getindex_value(b)sum(a)
    end
end

dot(a::AbstractFillVector, b::AbstractFillVector) = _fill_dot(a, b)
dot(a::AbstractFillVector, b::AbstractVector) = _fill_dot(a, b)
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
for TYPE in (:AbstractArray, :AbstractFill) # AbstractFill for disambiguity
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

# necessary for AbstractRange, Diagonal, etc
+(a::AbstractFill, b::AbstractFill) = fill_add(a, b)
+(a::AbstractFill, b::AbstractArray) = fill_add(b, a)
+(a::AbstractArray, b::AbstractFill) = fill_add(a, b)
-(a::AbstractFill, b::AbstractFill) = a + (-b)
-(a::AbstractFill, b::AbstractArray) = a + (-b)
-(a::AbstractArray, b::AbstractFill) = a + (-b)

@inline function fill_add(a, b::AbstractFill)
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
