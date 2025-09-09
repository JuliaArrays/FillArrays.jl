## vec

vec(a::AbstractFill) = fillsimilar(a, length(a))

## Transpose/Adjoint
# cannot do this for vectors since that would destroy scalar dot product


for OP in (:transpose, :adjoint)
    @eval begin
        function $OP(a::AbstractZerosMatrix)
            v = getindex_value(a)
            T = typeof($OP(v))
            Zeros{T}(reverse(axes(a)))
        end
        $OP(a::AbstractOnesMatrix) = fillsimilar(a, reverse(axes(a)))
        $OP(a::FillMatrix) = Fill($OP(a.value), reverse(a.axes))
    end
end

permutedims(a::AbstractFillVector) = fillsimilar(a, (1, length(a)))
permutedims(a::AbstractFillMatrix) = fillsimilar(a, reverse(axes(a)))

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

# Default outputs, can overload to customize
mult_fill(a, b, val, ax) = Fill(val, ax)
mult_zeros(a, b, elt, ax) = Zeros{elt}(ax)
mult_ones(a, b, elt, ax) = Ones{elt}(ax)

function mult_fill(a::AbstractFill, b::AbstractFill, ax)
    val = getindex_value(a)*getindex_value(b)*size(a,2)
    return mult_fill(a, b, val, ax)
end

function mult_zeros(a, b, ax)
    # This is currently only used in contexts where zero is defined
    # might need a rethink
    elt = typeof(zero(eltype(a)) * zero(eltype(b)))
    return mult_zeros(a, b, elt, ax)
end

function mult_ones(a, b, ax)
    # This is currently only used in contexts where zero is defined
    # might need a rethink
    elt = typeof(zero(eltype(a)) * zero(eltype(b)))
    return mult_ones(a, b, elt, ax)
end

function mult_axes(a, b)
    Base.require_one_based_indexing(a, b)
    size(a, 2) ≠ size(b, 1) &&
        throw(DimensionMismatch("A has dimensions $(size(a)) but B has dimensions $(size(b))"))
    return (axes(a, 1), axes(b)[2:end]...)
end

mult_fill(a, b) = mult_fill(a, b, mult_axes(a, b))
# for arrays of numbers, we assume that zero is defined for the result
# in this case, we may express the result as a Zeros
mult_zeros(a::AbstractArray{<:Number}, b::AbstractArray{<:Number}) = mult_zeros(a, b, mult_axes(a, b))
# In general, we create a Fill that doesn't assume anything about the
# properties of the element type
mult_zeros(a, b) = mult_fill(a, b, mult_axes(a, b))
mult_ones(a, b) = mult_ones(a, b, mult_axes(a, b))

*(a::AbstractFillMatrix, b::AbstractFillMatrix) = mult_fill(a,b)
*(a::AbstractFillMatrix, b::AbstractFillVector) = mult_fill(a,b)

# this treats a size (n,) vector as a nx1 matrix, so b needs to have 1 row
# special cased, as OnesMatrix * OnesMatrix isn't a Ones
*(a::AbstractOnesVector, b::AbstractOnesMatrix) = mult_ones(a, b)

*(a::AbstractZerosMatrix, b::AbstractZerosMatrix) = mult_zeros(a, b)
*(a::AbstractZerosMatrix, b::AbstractZerosVector) = mult_zeros(a, b)

*(a::AbstractZerosMatrix, b::AbstractFillMatrix) = mult_zeros(a, b)
*(a::AbstractZerosMatrix, b::AbstractFillVector) = mult_zeros(a, b)
*(a::AbstractFillMatrix, b::AbstractZerosMatrix) = mult_zeros(a, b)
*(a::AbstractFillMatrix, b::AbstractZerosVector) = mult_zeros(a, b)

for MT in (:AbstractMatrix, :AbstractTriangular)
    @eval *(a::AbstractZerosMatrix, b::$MT) = mult_zeros(a, b)
    @eval *(a::$MT, b::AbstractZerosMatrix) = mult_zeros(a, b)
end
# Odd way to deal with the type-parameters to avoid ambiguities
for MT in (:(AbstractMatrix{T}), :(Transpose{<:Any, <:AbstractMatrix{T}}), :(Adjoint{<:Any, <:AbstractMatrix{T}}),
            :(AbstractTriangular{T}))
    @eval *(a::$MT, b::AbstractZerosVector) where {T} = mult_zeros(a, b)
end
for T in (:AbstractZerosMatrix, :AbstractFillMatrix)
    @eval begin
        *(a::Transpose{<:Any, <:AbstractVector}, b::$T) = transpose(transpose(b) * parent(a))
        *(a::Adjoint{<:Any, <:AbstractVector}, b::$T) = adjoint(adjoint(b) * parent(a))
    end
end
*(a::AbstractZerosMatrix, b::AbstractVector) = mult_zeros(a, b)
function *(F::AbstractFillMatrix, v::AbstractVector)
    check_matmul_sizes(F, v)
    Fill(getindex_value(F) * sum(v), (axes(F,1),))
end

function lmul_diag(a::Diagonal, b)
    size(a,2) == size(b,1) || throw(DimensionMismatch("A has dimensions $(size(a)) but B has dimensions $(size(b))"))
    parent(a) .* b # use special broadcast
end
function rmul_diag(a, b::Diagonal)
    size(a,2) == size(b,1) || throw(DimensionMismatch("A has dimensions $(size(a)) but B has dimensions $(size(b))"))
    a .* permutedims(parent(b)) # use special broadcast
end

*(a::AbstractZerosMatrix, b::Diagonal) = rmul_diag(a, b)
*(a::Diagonal, b::AbstractZerosVector) = lmul_diag(a, b)
*(a::Diagonal, b::AbstractZerosMatrix) = lmul_diag(a, b)
*(a::Diagonal, b::AbstractFillMatrix) = lmul_diag(a, b)
*(a::AbstractFillMatrix, b::Diagonal) = rmul_diag(a, b)

@noinline function check_matmul_sizes(A::AbstractMatrix, x::AbstractVector)
    Base.require_one_based_indexing(A, x)
    size(A,2) == size(x,1) ||
        throw(DimensionMismatch("second dimension of A, $(size(A,2)) does not match length of x $(length(x))"))
end
@noinline function check_matmul_sizes(A::AbstractMatrix, B::AbstractMatrix)
    Base.require_one_based_indexing(A, B)
    size(A,2) == size(B,1) ||
        throw(DimensionMismatch("second dimension of A, $(size(A,2)) does not match first dimension of B, $(size(B,1))"))
end
@noinline function check_matmul_sizes(y::AbstractVector, A::AbstractMatrix, x::AbstractVector)
    Base.require_one_based_indexing(A, x, y)
    size(A,2) == size(x,1) ||
        throw(DimensionMismatch("second dimension of A, $(size(A,2)) does not match length of x $(length(x))"))
    size(y,1) == size(A,1) ||
        throw(DimensionMismatch("first dimension of A, $(size(A,1)) does not match length of y $(length(y))"))
end
@noinline function check_matmul_sizes(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    Base.require_one_based_indexing(A, B, C)
    size(A,2) == size(B,1) ||
        throw(DimensionMismatch("second dimension of A, $(size(A,2)) does not match first dimension of B, $(size(B,1))"))
    size(C,1) == size(A,1) && size(C,2) == size(B,2) ||
        throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B)), C has size $(size(C))"))
end

function mul!(y::AbstractVector, A::AbstractFillMatrix, b::AbstractFillVector, alpha::Number, beta::Number)
    check_matmul_sizes(y, A, b)

    Abα = Ref(getindex_value(A) * getindex_value(b) * alpha * length(b))

    if iszero(beta)
        y .= Abα
    else
        y .= Abα .+ y .* beta
    end
    y
end

function mul!(y::StridedVector, A::StridedMatrix, b::AbstractFillVector, alpha::Number, beta::Number)
    check_matmul_sizes(y, A, b)

    bα = Ref(getindex_value(b) * alpha)

    if iszero(beta)
        y .= Ref(zero(eltype(y)))
    else
        rmul!(y, beta)
    end
    for Acol in eachcol(A)
        @. y += Acol * bα
    end
    y
end

function mul!(y::StridedVector, A::AbstractFillMatrix, b::StridedVector, alpha::Number, beta::Number)
    check_matmul_sizes(y, A, b)

    Abα = Ref(getindex_value(A) * sum(b) * alpha)

    if iszero(beta)
        y .= Abα
    else
        y .= Abα .+ y .* beta
    end
    y
end

function _mul_adjtrans!(y::AbstractVector, A::AbstractMatrix, b::AbstractFillVector, alpha, beta, f)
    bα = getindex_value(b) * alpha
    At = f(A)

    if iszero(beta)
        for (ind, Atcol) in zip(eachindex(y), eachcol(At))
            y[ind] = f(sum(Atcol)) * bα
        end
    else
        for (ind, Atcol) in zip(eachindex(y), eachcol(At))
            y[ind] = f(sum(Atcol)) * bα .+ y[ind] .* beta
        end
    end
    y
end

for (T, f) in ((:Adjoint, :adjoint), (:Transpose, :transpose))
    @eval function mul!(y::StridedVector, A::$T{<:Any, <:StridedMatrix}, b::AbstractFillVector, alpha::Number, beta::Number)
        check_matmul_sizes(y, A, b)
        _mul_adjtrans!(y, A, b, alpha, beta, $f)
    end
end

# unnecessary indirection, added for ambiguity resolution
function _mulfill!(C::AbstractMatrix, A::AbstractFillMatrix, B::AbstractFillMatrix, alpha, beta)
    check_matmul_sizes(C, A, B)
    ABα = getindex_value(A) * getindex_value(B) * alpha * size(B,1)
    if iszero(beta)
        C .= ABα
    else
        C .= ABα .+ C .* beta
    end
    return C
end

function mul!(C::AbstractMatrix, A::AbstractFillMatrix, B::AbstractFillMatrix, alpha::Number, beta::Number)
    _mulfill!(C, A, B, alpha, beta)
    return C
end

function copyfirstcol!(C)
    @views for i in axes(C,2)[2:end]
        C[:, i] .= C[:, 1]
    end
    return C
end

_firstcol(C::AbstractMatrix) = first(eachcol(C))

function copyfirstrow!(C)
    # C[begin+1:end, ind] .= permutedims(_firstrow(C))
    # we loop here as the aliasing check isn't smart enough to
    # detect that the two sides don't alias, and ends up materializing the RHS
    for (ind, v) in pairs(_firstrow(C))
        C[begin+1:end, ind] .= Ref(v)
    end
    return C
end
_firstrow(C::AbstractMatrix) = first(eachrow(C))

function _mulfill!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractFillMatrix, alpha, beta)
    check_matmul_sizes(C, A, B)
    iszero(size(B,2)) && return C # no columns in B and C, empty matrix
    if iszero(beta)
        # the mat-vec product sums along the rows of A
        mul!(_firstcol(C), A, _firstcol(B), alpha, beta)
        copyfirstcol!(C)
    else
        # the mat-vec product sums along the rows of A, which produces the first column of ABα
        # allocate a temporary column vector to store the result
        v = A * (_firstcol(B) * alpha)
        C .= v .+ C .* beta
    end
    return C
end
function _mulfill!(C::AbstractMatrix, A::AbstractFillMatrix, B::AbstractMatrix, alpha, beta)
    check_matmul_sizes(C, A, B)
    iszero(size(A,1)) && return C # no rows in A and C, empty matrix
    Aval = getindex_value(A)
    if iszero(beta)
        Crow = _firstrow(C)
        # sum along the columns of B
        Crow .= Ref(Aval) .* sum.(eachcol(B)) .* alpha
        copyfirstrow!(C)
    else
        # sum along the columns of B, and allocate the result.
        # This is the first row of ABα
        ABα_row = Ref(Aval) .* sum.(eachcol(B)) .* alpha
        C .= permutedims(ABα_row) .+ C .* beta
    end
    return C
end

function mul!(C::StridedMatrix, A::StridedMatrix, B::AbstractFillMatrix, alpha::Number, beta::Number)
    _mulfill!(C, A, B, alpha, beta)
    return C
end
function mul!(C::StridedMatrix, A::AbstractFillMatrix, B::StridedMatrix, alpha::Number, beta::Number)
    _mulfill!(C, A, B, alpha, beta)
    return C
end

for T in (:Adjoint, :Transpose)
    @eval begin
        function mul!(C::StridedMatrix, A::$T{<:Any, <:StridedMatrix}, B::AbstractFillMatrix, alpha::Number, beta::Number)
            _mulfill!(C, A, B, alpha, beta)
            return C
        end
        function mul!(C::StridedMatrix, A::AbstractFillMatrix, B::$T{<:Any, <:StridedMatrix}, alpha::Number, beta::Number)
            _mulfill!(C, A, B, alpha, beta)
            return C
        end
    end
end

function _adjvec_mul_zeros(a, b)
    la, lb = length(a), length(b)
    if la ≠ lb
        throw(DimensionMismatch("dot product arguments have lengths $la and $lb"))
    end
    # ensure that all the elements of `a` are of the same size,
    # so that ∑ᵢaᵢbᵢ = b₁∑ᵢaᵢ makes sense
    if la == 0
        # this errors if a is a nested array, and zero isn't well-defined
        return zero(eltype(a)) * zero(eltype(b))
    end
    a1 = a[1]
    sza1 = size(a1)
    all(x -> size(x) == sza1, a) || throw(DimensionMismatch("not all elements of A are of size $sza1"))
    # we replace b₁∑ᵢaᵢ by b₁a₁, as we know that b₁ is zero.
    # Each term in the summation is zero, so the sum is equal to the first term
    return a1 * b[1]
end

for MT in (:AbstractMatrix, :AbstractTriangular, :(Adjoint{<:Any,<:TransposeAbsVec}), :AbstractFillMatrix)
    @eval *(a::AdjointAbsVec{<:Any,<:AbstractZerosVector}, b::$MT) = (b' * a')'
end
# ambiguity
function *(a::AdjointAbsVec{<:Any,<:AbstractZerosVector}, b::TransposeAbsVec{<:Any,<:AdjointAbsVec})
    # change from Transpose ∘ Adjoint to Adjoint ∘ Transpose
    b2 = adjoint(transpose(adjoint(transpose(b))))
    a * b2
end
*(a::AdjointAbsVec{<:Any,<:AbstractZerosVector}, b::AbstractZerosMatrix) = (b' * a')'
for MT in (:AbstractMatrix, :AbstractTriangular, :(Transpose{<:Any,<:AdjointAbsVec}), :AbstractFillMatrix)
    @eval *(a::TransposeAbsVec{<:Any,<:AbstractZerosVector}, b::$MT) = transpose(transpose(b) * transpose(a))
end
*(a::TransposeAbsVec{<:Any,<:AbstractZerosVector}, b::AbstractZerosMatrix) = transpose(transpose(b) * transpose(a))

*(a::AbstractVector, b::AdjOrTransAbsVec{<:Any,<:AbstractZerosVector}) = a * permutedims(parent(b))
for MT in (:AbstractMatrix, :AbstractTriangular)
    @eval *(a::$MT, b::AdjOrTransAbsVec{<:Any,<:AbstractZerosVector}) = a * permutedims(parent(b))
end
*(a::AbstractZerosVector, b::AdjOrTransAbsVec{<:Any,<:AbstractZerosVector}) = a * permutedims(parent(b))
*(a::AbstractZerosMatrix, b::AdjOrTransAbsVec{<:Any,<:AbstractZerosVector}) = a * permutedims(parent(b))

*(a::AdjointAbsVec, b::AbstractZerosVector) = _adjvec_mul_zeros(a, b)
*(a::AdjointAbsVec{<:Number}, b::AbstractZerosVector{<:Number}) = _adjvec_mul_zeros(a, b)
*(a::TransposeAbsVec, b::AbstractZerosVector) = _adjvec_mul_zeros(a, b)
*(a::TransposeAbsVec{<:Number}, b::AbstractZerosVector{<:Number}) = _adjvec_mul_zeros(a, b)

*(a::Adjoint{T, <:AbstractMatrix{T}} where T, b::AbstractZeros{<:Any, 1}) = mult_zeros(a, b)

*(D::Diagonal, a::Adjoint{<:Any,<:AbstractZerosVector}) = (a' * D')'
*(D::Diagonal, a::Transpose{<:Any,<:AbstractZerosVector}) = transpose(transpose(a) * transpose(D))
*(a::AdjointAbsVec{<:Any,<:AbstractZerosVector}, D::Diagonal) = (D' * a')'
*(a::TransposeAbsVec{<:Any,<:AbstractZerosVector}, D::Diagonal) = transpose(D*transpose(a))
function _triple_zeromul(x, D::Diagonal, y)
    if !(length(x) == length(D.diag) == length(y))
        throw(DimensionMismatch("x has length $(length(x)), D has size $(size(D)), and y has $(length(y))"))
    end
    zero(promote_type(eltype(x), eltype(D), eltype(y)))
end

*(x::AdjointAbsVec{<:Any,<:AbstractZerosVector}, D::Diagonal, y::AbstractVector) = _triple_zeromul(x, D, y)
*(x::TransposeAbsVec{<:Any,<:AbstractZerosVector}, D::Diagonal, y::AbstractVector) = _triple_zeromul(x, D, y)
*(x::AdjointAbsVec, D::Diagonal, y::AbstractZerosVector) = _triple_zeromul(x, D, y)
*(x::TransposeAbsVec, D::Diagonal, y::AbstractZerosVector) = _triple_zeromul(x, D, y)
*(x::AdjointAbsVec{<:Any,<:AbstractZerosVector}, D::Diagonal, y::AbstractZerosVector) = _triple_zeromul(x, D, y)
*(x::TransposeAbsVec{<:Any,<:AbstractZerosVector}, D::Diagonal, y::AbstractZerosVector) = _triple_zeromul(x, D, y)


function *(a::Transpose{T, <:AbstractVector}, b::AbstractZerosVector{T}) where T<:Real
    la, lb = length(a), length(b)
    if la ≠ lb
        throw(DimensionMismatch("dot product arguments have lengths $la and $lb"))
    end
    return zero(T)
end
*(a::Transpose{T, <:AbstractMatrix{T}}, b::AbstractZerosVector{T}) where T<:Real = mult_zeros(a, b)

# support types with fast sum
# infinite cases should be supported in InfiniteArrays.jl
# type issues of Bool dot are ignored at present.
function _fill_dot(a::AbstractFillVector{T}, b::AbstractVector{V}) where {T,V}
    axes(a) == axes(b) || throw(DimensionMismatch("dot product arguments have lengths $(length(a)) and $(length(b))"))
    dot(getindex_value(a), sum(b))
end

function _fill_dot_rev(a::AbstractVector{T}, b::AbstractFillVector{V}) where {T,V}
    axes(a) == axes(b) || throw(DimensionMismatch("dot product arguments have lengths $(length(a)) and $(length(b))"))
    dot(sum(a), getindex_value(b))
end

dot(a::AbstractFillVector, b::AbstractFillVector) = _fill_dot(a, b)
dot(a::AbstractFillVector, b::AbstractVector) = _fill_dot(a, b)
dot(a::AbstractVector, b::AbstractFillVector) = _fill_dot_rev(a, b)

function dot(u::AbstractVector, E::Eye, v::AbstractVector)
    length(u) == size(E,1) && length(v) == size(E,2) ||
        throw(DimensionMismatch("dot product arguments have dimensions $(length(u))×$(size(E))×$(length(v))"))
    d = dot(u,v)
    T = typeof(one(eltype(E)) * d)
    convert(T, d)
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
+(a::AbstractFill) = a
-(a::AbstractZeros) = a
-(a::AbstractFill) = Fill(-getindex_value(a), size(a))

# special-cased for type-stability, as Ones + Ones is not a Ones
Base.reduce_first(::typeof(+), x::AbstractOnes) = Fill(Base.reduce_first(+, getindex_value(x)), axes(x))

function +(a::AbstractZeros{T}, b::AbstractZeros{V}) where {T, V} # for disambiguity
    promote_shape(a,b)
    return elconvert(promote_op(+,T,V),a)
end
# no AbstractArray. Otherwise incompatible with StaticArrays.jl
# AbstractFill for disambiguity
for TYPE in (:Array, :AbstractFill, :AbstractRange, :Diagonal)
    @eval function +(a::$TYPE{T}, b::AbstractZeros{V}) where {T, V}
        promote_shape(a,b)
        return elconvert(promote_op(+,T,V),a)
    end
    @eval +(a::AbstractZeros, b::$TYPE) = b + a
end

# for VERSION other than 1.6, could use ZerosMatrix only
function +(a::AbstractFillMatrix{T}, b::UniformScaling) where {T}
    n = checksquare(a)
    return a + Diagonal(Fill(zero(T) + b.λ, n))
end

# LinearAlgebra defines `-(a::AbstractMatrix, b::UniformScaling) = a + (-b)`,
# so the implementation of `-(a::UniformScaling, b::AbstractFill{<:Any,2})` is sufficient
-(a::UniformScaling, b::AbstractFill) = -b + a # @test I-Zeros(3,3) === Diagonal(Ones(3))

# TODO: How to do this conversion generically?
-(a::AbstractOnes, b::AbstractOnes) = broadcasted_zeros(+, a, eltype(a), axes(a)) + broadcasted_zeros(-, b, eltype(a), axes(a))

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
    a .+ (getindex_value(b),)
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
    @eval $op(a::AbstractZeros) = norm(getindex_value(a))
end

normp(a::AbstractZeros, p) = norm(getindex_value(a))

norm1(a::AbstractFill) = length(a)*norm(getindex_value(a))
function norm2(a::AbstractFill)
    nrm1 = norm(getindex_value(a))
    sqrt(oftype(nrm1, length(a)))*nrm1
end
function normp(a::AbstractFill, p)
    nrm1 = norm(getindex_value(a))
    (length(a))^(1/oftype(nrm1, p))*nrm1
end
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
fillzero(::Type{<:AbstractZeros{T,N,AXIS}}, n, m) where {T,N,AXIS} = Zeros{T,N,AXIS}((n, m))
fillzero(::Type{F}, n, m) where F = throw(ArgumentError("Cannot create a zero array of type $F"))

diagzero(D::Diagonal{F}, i, j) where F<:AbstractFill = fillzero(F, axes(D.diag[i], 1), axes(D.diag[j], 2))

# kron

# Default outputs, can overload to customize
kron_fill(a, b, val, ax) = Fill(val, ax)
kron_zeros(a, b, elt, ax) = Zeros{elt}(ax)
kron_ones(a, b, elt, ax) = Ones{elt}(ax)

_kronsize(f::AbstractFillVector, g::AbstractFillVector) = (size(f,1)*size(g,1),)
_kronsize(f::AbstractFillVecOrMat, g::AbstractFillVecOrMat) = (size(f,1)*size(g,1), size(f,2)*size(g,2))
function _kron(f::AbstractFill, g::AbstractFill, sz)
    v = getindex_value(f)*getindex_value(g)
    return kron_fill(f, g, v, sz)
end
function _kron(f::AbstractZeros, g::AbstractZeros, sz)
    elt = promote_type(eltype(f), eltype(g))
    return kron_zeros(f, g, elt, sz)
end
function _kron(f::AbstractOnes, g::AbstractOnes, sz)
    elt = promote_type(eltype(f), eltype(g))
    return kron_ones(f, g, elt, sz)
end
function kron(f::AbstractFillVecOrMat, g::AbstractFillVecOrMat)
    sz = _kronsize(f, g)
    return _kron(f, g, sz)
end

# bandedness
function LinearAlgebra.istriu(A::AbstractFillMatrix, k::Integer = 0)
    iszero(A) || k <= -(size(A,1)-1)
end
function LinearAlgebra.istril(A::AbstractFillMatrix, k::Integer = 0)
    iszero(A) || k >= size(A,2)-1
end

triu(A::AbstractZerosMatrix, k::Integer=0) = A
tril(A::AbstractZerosMatrix, k::Integer=0) = A
