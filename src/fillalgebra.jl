## vec

vec(a::AbstractFill) = fillsimilar(a, length(a))

## Transpose/Adjoint
# cannot do this for vectors since that would destroy scalar dot product


transpose(a::Union{OnesMatrix, ZerosMatrix}) = fillsimilar(a, reverse(axes(a)))
adjoint(a::Union{OnesMatrix, ZerosMatrix}) = fillsimilar(a, reverse(axes(a)))
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

function _mult_fill(a::AbstractFill, b::AbstractFill, ax, ::Type{Fill})
    val = getindex_value(a)*getindex_value(b)*size(a,2)
    return Fill(val, ax)
end

function _mult_fill(a, b, ax, ::Type{OnesZeros}) where {OnesZeros}
    ElType = promote_type(eltype(a), eltype(b))
    return OnesZeros{ElType}(ax)
end

function mult_fill(a, b, T::Type = Fill)
    Base.require_one_based_indexing(a, b)
    size(a, 2) ≠ size(b, 1) &&
        throw(DimensionMismatch("A has dimensions $(size(a)) but B has dimensions $(size(b))"))
    ax_result = (axes(a, 1), axes(b)[2:end]...)
    _mult_fill(a, b, ax_result, T)
end
mult_zeros(a, b) = mult_fill(a, b, Zeros)
mult_ones(a, b) = mult_fill(a, b, Ones)

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
*(a::AbstractFillVector, b::ZerosMatrix) = mult_zeros(a, b)
*(a::AbstractFillMatrix, b::ZerosMatrix) = mult_zeros(a, b)
*(a::AbstractFillMatrix, b::ZerosVector) = mult_zeros(a, b)

*(a::ZerosVector, b::AbstractMatrix) = mult_zeros(a, b)
*(a::ZerosMatrix, b::AbstractMatrix) = mult_zeros(a, b)
*(a::AbstractMatrix, b::ZerosVector) = mult_zeros(a, b)
*(a::AbstractMatrix, b::ZerosMatrix) = mult_zeros(a, b)
*(a::ZerosMatrix, b::AbstractVector) = mult_zeros(a, b)
*(a::AbstractVector, b::ZerosMatrix) = mult_zeros(a, b)

*(a::ZerosVector, b::AdjOrTransAbsVec) = mult_zeros(a, b)

*(a::ZerosVector, b::Diagonal) = mult_zeros(a, b)
*(a::ZerosMatrix, b::Diagonal) = mult_zeros(a, b)
*(a::Diagonal, b::ZerosVector) = mult_zeros(a, b)
*(a::Diagonal, b::ZerosMatrix) = mult_zeros(a, b)
function *(a::Diagonal, b::AbstractFillMatrix)
    size(a,2) == size(b,1) || throw(DimensionMismatch("A has dimensions $(size(a)) but B has dimensions $(size(b))"))
    parent(a) .* b # use special broadcast
end
function *(a::AbstractFillMatrix, b::Diagonal)
    size(a,2) == size(b,1) || throw(DimensionMismatch("A has dimensions $(size(a)) but B has dimensions $(size(b))"))
    a .* permutedims(parent(b)) # use special broadcast
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

    αAb = alpha * getindex_value(A) * getindex_value(b) * length(b)

    if iszero(beta)
        y .= αAb
    else
        y .= αAb .+ beta .* y
    end
    y
end

function mul!(y::StridedVector, A::StridedMatrix, b::AbstractFillVector, alpha::Number, beta::Number)
    check_matmul_sizes(y, A, b)

    αb = alpha * getindex_value(b)

    if iszero(beta)
        y .= zero(eltype(y))
        for col in eachcol(A)
            y .+= αb .* col
        end
    else
        lmul!(beta, y)
        for col in eachcol(A)
            y .+= αb .* col
        end
    end
    y
end

function mul!(y::StridedVector, A::AbstractFillMatrix, b::StridedVector, alpha::Number, beta::Number)
    check_matmul_sizes(y, A, b)

    αA = alpha * getindex_value(A)

    if iszero(beta)
        y .= αA .* sum(b)
    else
        y .= αA .* sum(b) .+ beta .* y
    end
    y
end

function _mul_adjtrans!(y::AbstractVector, A::AbstractMatrix, b::AbstractVector, alpha, beta, f)
    α = alpha * getindex_value(b)

    At = f(A)

    if iszero(beta)
        for (ind, col) in zip(eachindex(y), eachcol(At))
            y[ind] = α .* f(sum(col))
        end
    else
        for (ind, col) in zip(eachindex(y), eachcol(At))
            y[ind] = α .* f(sum(col)) .+ beta .* y[ind]
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

function mul!(C::AbstractMatrix, A::AbstractFillMatrix, B::AbstractFillMatrix, alpha::Number, beta::Number)
    check_matmul_sizes(C, A, B)
    αAB = alpha * getindex_value(A) * getindex_value(B) * size(B,1)
    if iszero(beta)
        C .= αAB
    else
        C .= αAB .+ beta .* C
    end
    C
end

function copyfirstcol!(C)
    @views for i in axes(C,2)[2:end]
        C[:, i] .= C[:, 1]
    end
    return C
end
function copyfirstcol!(C::Union{Adjoint, Transpose})
    # in this case, we copy the first row of the parent to others
    Cp = parent(C)
    for colind in axes(Cp, 2)
        Cp[2:end, colind] .= Cp[1, colind]
    end
    return C
end

_firstcol(C::AbstractMatrix) = view(C, :, 1)
_firstcol(C::Union{Adjoint, Transpose}) = view(parent(C), 1, :)

function _mulfill!(C, A, B::AbstractFillMatrix, alpha, beta)
    check_matmul_sizes(C, A, B)
    if iszero(size(B,2))
        return lmul!(beta, C)
    end
    mul!(_firstcol(C), A, view(B, :, 1), alpha, beta)
    copyfirstcol!(C)
    C
end

function mul!(C::StridedMatrix, A::StridedMatrix, B::AbstractFillMatrix, alpha::Number, beta::Number)
    _mulfill!(C, A, B, alpha, beta)
end

for T in (:Adjoint, :Transpose)
    @eval function mul!(C::StridedMatrix, A::$T{<:Any, <:StridedMatrix}, B::AbstractFillMatrix, alpha::Number, beta::Number)
        _mulfill!(C, A, B, alpha, beta)
    end
end

function mul!(C::StridedMatrix, A::AbstractFillMatrix, B::StridedMatrix, alpha::Number, beta::Number)
    check_matmul_sizes(C, A, B)
    for (colC, colB) in zip(eachcol(C), eachcol(B))
        mul!(colC, A, colB, alpha, beta)
    end
    C
end

for (T, f) in ((:Adjoint, :adjoint), (:Transpose, :transpose))
    @eval function mul!(C::StridedMatrix, A::AbstractFillMatrix, B::$T{<:Any, <:StridedMatrix}, alpha::Number, beta::Number)
        _mulfill!($f(C), $f(B), $f(A), alpha, beta)
        C
    end
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
+(a::AbstractFill) = a
-(a::Zeros) = a
-(a::AbstractFill) = Fill(-getindex_value(a), size(a))


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

_eigenind(λ0, n, sortby) = (isnothing(sortby) || sortby(λ0) <= sortby(zero(λ0))) ? 1 : n

function eigvals(A::AbstractFillMatrix{<:Number}; sortby = nothing)
    Base.require_one_based_indexing(A)
    n = checksquare(A)
    # only one non-trivial eigenvalue for a rank-1 matrix
    λ0 = float(getindex_value(A)) * n
    ind = _eigenind(λ0, n, sortby)
    OneElement(λ0, ind, n)
end

function eigvecs(A::AbstractFillMatrix{<:Number}; sortby = nothing)
    Base.require_one_based_indexing(A)
    n = checksquare(A)
    M = similar(A, float(eltype(A)))
    n == 0 && return M
    val = getindex_value(A)
    ind = _eigenind(val, n, sortby)
    # The non-trivial eigenvector is normalize(ones(n))
    M[:, ind] .= 1/sqrt(n)
    # eigenvectors corresponding to zero eigenvalues
    for (i, j) in enumerate(axes(M,2)[(ind == 1) .+ (1:end-1)])
        # The eigenvectors are v = normalize([ones(n-1); -(n-1)]), and sum(v) == 0
        # The ordering is arbitrary,
        # and we choose to order in terms of the number of non-zero elements
        nrm = 1/sqrt(i*(i+1))
        M[1:i, j] .= nrm
        M[i+1, j] = -i * nrm
        M[i+2:end, j] .= zero(eltype(M))
    end
    return M
end

function eigen(A::AbstractFillMatrix{<:Number}; sortby = nothing)
    Eigen(eigvals(A; sortby), eigvecs(A; sortby))
end

# Tridiagonal Toeplitz eigen following Trench (1985)
# "On the eigenvalue problem for Toeplitz band matrices"
# https://www.sciencedirect.com/science/article/pii/0024379585902770

for MT in (:(Tridiagonal{<:Union{Real, Complex}, <:AbstractFillVector}),
            :(SymTridiagonal{<:Union{Real, Complex}, <:AbstractFillVector}),
            :(HermOrSym{T, <:Tridiagonal{T, <:AbstractFillVector{T}}} where {T<:Union{Real, Complex}})
            )
    @eval function eigvals(A::$MT; sortby = nothing)
        _eigvals_toeplitz(A; sortby)
    end
end

___eigvals_toeplitz(a, sqrtbc, n) = [a + 2 * sqrtbc * cospi(q/(n+1)) for q in n:-1:1]

__eigvals_toeplitz(::AbstractMatrix, a, b, c, n) =
    ___eigvals_toeplitz(a, √(b*c), n)
__eigvals_toeplitz(::Union{SymTridiagonal, Symmetric{<:Any, <:Tridiagonal}}, a, b, c, n) =
    ___eigvals_toeplitz(a, b, n)
__eigvals_toeplitz(::Hermitian{<:Any, <:Tridiagonal}, a, b, c, n) =
    ___eigvals_toeplitz(real(a), abs(b), n)

# triangular Toeplitz
function _eigvals_toeplitz(T; sortby = nothing)
    Base.require_one_based_indexing(T)
    n = checksquare(T)
    # extra care to handle 0x0 and 1x1 matrices
    # diagonal
    a = get(T, (1,1), zero(eltype(T)))
    # subdiagonal
    b = get(T, (2,1), zero(eltype(T)))
    # superdiagonal
    c = get(T, (1,2), zero(eltype(T)))
    vals = __eigvals_toeplitz(T, a, b, c, n)
    !isnothing(sortby) && sort!(vals, by = sortby)
    return vals
end

_eigvec_prefactor(A, cm1, c1, m) = sqrt(complex(cm1/c1))^m
_eigvec_prefactor(A::Union{SymTridiagonal, Symmetric{<:Any, <:Tridiagonal}}, cm1, c1, m) = oneunit(_eigvec_eltype(A))

function _eigvec_prefactors(A, cm1, c1)
    x = _eigvec_prefactor(A, cm1, c1, 1)
    [x^(j-1) for j in axes(A,1)]
end
_eigvec_prefactors(A::Union{SymTridiagonal, Symmetric{<:Any, <:Tridiagonal}}, cm1, c1) =
    Fill(_eigvec_prefactor(A, cm1, c1, 1), size(A,1))

_eigvec_eltype(A::SymTridiagonal) = float(eltype(A))
_eigvec_eltype(A) = complex(float(eltype(A)))

# The functions swapcols! and permutecols!! are copied from Julia Base, which is under the MIT License
# https://github.com/JuliaLang/julia/blob/master/LICENSE.md
#=
MIT License
Copyright (c) 2009-2022: Jeff Bezanson, Stefan Karpinski, Viral B. Shah,
and other contributors: https://github.com/JuliaLang/julia/contributors
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.
=#
# swap columns i and j of a, in-place
function swapcols!(a::AbstractMatrix, i, j)
    i == j && return
    cols = axes(a,2)
    @boundscheck i in cols || throw(BoundsError(a, (:,i)))
    @boundscheck j in cols || throw(BoundsError(a, (:,j)))
    for k in axes(a,1)
        @inbounds a[k,i],a[k,j] = a[k,j],a[k,i]
    end
end

function permutecols!!(A::AbstractMatrix, p::AbstractVector{<:Integer})
    Base.require_one_based_indexing(A, p)
    count = 0
    start = 0
    while count < length(p)
        ptr = start = findnext(!iszero, p, start+1)
        next = p[start]
        count += 1
        while next != start
            swapcols!(A, ptr, next)
            p[ptr] = 0
            ptr = next
            next = p[next]
            count += 1
        end
        p[ptr] = 0
    end
    A
end

function _eigvecs_toeplitz(T; sortby = nothing)
    Base.require_one_based_indexing(T)
    n = checksquare(T)
    M = Matrix{_eigvec_eltype(T)}(undef, n, n)
    n == 0 && return M
    n == 1 && return fill!(M, oneunit(eltype(M)))
    cm1 = T[2,1] # subdiagonal
    c1 = T[1,2]  # superdiagonal
    prefactors = _eigvec_prefactors(T, cm1, c1)
    for q in axes(M,2)
        qrev = n+1-q # match the default eigenvalue sorting
        for j in axes(M,1)
            M[j, q] = prefactors[j] * sinpi(j*qrev/(n+1))
        end
        normalize!(view(M, :, q))
    end
    if !isnothing(sortby)
        perm = sortperm(eigvals(T), by = sortby)
        permutecols!!(M, perm)
    end
    return M
end

for MT in (:(Tridiagonal{<:Union{Real,Complex}, <:AbstractFillVector}),
            :(SymTridiagonal{<:Union{Real,Complex}, <:AbstractFillVector}),
            :(HermOrSym{T, <:Tridiagonal{T, <:AbstractFillVector{T}}} where {T<:Union{Real,Complex}}),
            )

    @eval begin
        function eigvecs(A::$MT; sortby = nothing)
            _eigvecs_toeplitz(A; sortby)
        end
        function eigen(T::$MT; sortby = nothing)
            Eigen(eigvals(T; sortby), eigvecs(T; sortby))
        end
    end
end


