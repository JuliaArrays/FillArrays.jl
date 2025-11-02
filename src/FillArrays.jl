""" `FillArrays` module to lazily represent matrices with a single value """
module FillArrays

using LinearAlgebra
import Base: size, getindex, setindex!, IndexStyle, checkbounds, convert,
    +, -, *, /, \, diff, sum, cumsum, maximum, minimum, sort, sort!,
    any, all, axes, isone, iszero, iterate, unique, allunique, permutedims, inv,
    copy, vec, setindex!, count, ==, reshape, map, zero,
    show, view, in, mapreduce, one, reverse, promote_op, promote_rule, repeat,
    parent, similar, issorted, add_sum, accumulate, OneTo, permutedims,
    real, imag, conj

import LinearAlgebra: rank, svdvals!, tril, triu, tril!, triu!, diag, transpose, adjoint, fill!,
    dot, norm2, norm1, normInf, normMinusInf, normp, lmul!, rmul!, diagzero, AdjointAbsVec, TransposeAbsVec,
    issymmetric, ishermitian, AdjOrTransAbsVec, checksquare, mul!, kron, AbstractTriangular


import Base.Broadcast: broadcasted, DefaultArrayStyle, broadcast_shape, BroadcastStyle, Broadcasted

export Zeros, Ones, Fill, Eye, Trues, Falses, OneElement

import Base: oneto

"""
    AbstractFill{T, N, Axes} <: AbstractArray{T, N}

Supertype for lazy array types whose entries are all equal.
Subtypes of `AbstractFill` should implement [`FillArrays.getindex_value`](@ref) to return the value of the entries.
"""
abstract type AbstractFill{T, N, Axes} <: AbstractArray{T, N} end
const AbstractFillVector{T} = AbstractFill{T,1}
const AbstractFillMatrix{T} = AbstractFill{T,2}
const AbstractFillVecOrMat{T} = Union{AbstractFillVector{T},AbstractFillMatrix{T}}

==(a::AbstractFill, b::AbstractFill) = axes(a) == axes(b) && getindex_value(a) == getindex_value(b)

@inline function Base.isassigned(F::AbstractFill, i::Integer...)
    @boundscheck checkbounds(Bool, F, to_indices(F, i)...) || return false
    return true
end

@inline function _fill_getindex(F::AbstractFill, kj::Integer...)
    @boundscheck checkbounds(F, kj...)
    getindex_value(F)
end

Base.@propagate_inbounds getindex(F::AbstractFill, k::Integer) = _fill_getindex(F, k)
Base.@propagate_inbounds getindex(F::AbstractFill{T, N}, kj::Vararg{Integer, N}) where {T, N} = _fill_getindex(F, kj...)

@inline function setindex!(F::AbstractFill, v, k::Integer)
    @boundscheck checkbounds(F, k)
    v == getindex_value(F) || throw(ArgumentError("Cannot setindex! to $v for an AbstractFill with value $(getindex_value(F))."))
    F
end

@inline function setindex!(F::AbstractFill{T, N}, v, kj::Vararg{Integer, N}) where {T, N}
    @boundscheck checkbounds(F, kj...)
    v == getindex_value(F) || throw(ArgumentError("Cannot setindex! to $v for an AbstractFill with value $(getindex_value(F))."))
    F
end

@inline function fill!(F::AbstractFill, v)
    v == getindex_value(F) || throw(ArgumentError("Cannot fill! with $v an AbstractFill with value $(getindex_value(F))."))
    F
end

rank(F::AbstractFill) = iszero(getindex_value(F)) ? 0 : 1
IndexStyle(::Type{<:AbstractFill{<:Any,N,<:NTuple{N,Base.OneTo{Int}}}}) where N = IndexLinear()

issymmetric(F::AbstractFillMatrix) = axes(F,1) == axes(F,2) && (isempty(F) || issymmetric(getindex_value(F)))
ishermitian(F::AbstractFillMatrix) = axes(F,1) == axes(F,2) && (isempty(F) || ishermitian(getindex_value(F)))

Base.IteratorSize(::Type{<:AbstractFill{T,N,Axes}}) where {T,N,Axes} = _IteratorSize(Axes)
_IteratorSize(::Type{Tuple{}}) = Base.HasShape{0}()
_IteratorSize(::Type{Tuple{T}}) where {T} = Base.IteratorSize(T)
function _IteratorSize(::Type{T}) where {T<:Tuple}
    N = fieldcount(T)
    s = ntuple(i-> Base.IteratorSize(fieldtype(T, i)), N)
    any(x -> x isa Base.IsInfinite, s) ? Base.IsInfinite() : Base.HasShape{N}()
end


"""
    Fill{T, N, Axes} where {T,N,Axes<:Tuple{Vararg{AbstractUnitRange,N}}}

A lazy representation of an array of dimension `N`
whose entries are all equal to a constant of type `T`,
with axes of type `Axes`.
Typically created by `Fill` or `Zeros` or `Ones`

# Examples

```jldoctest
julia> Fill(7, (2,3))
2×3 Fill{Int64}, with entries equal to 7

julia> Fill{Float64, 1, Tuple{UnitRange{Int64}}}(7.0, (1:2,))
2-element Fill{Float64, 1, Tuple{UnitRange{Int64}}} with indices 1:2, with entries equal to 7.0
```
"""
struct Fill{T, N, Axes} <: AbstractFill{T, N, Axes}
    value::T
    axes::Axes

    Fill{T,N,Axes}(x::T, sz::Axes) where Axes<:Tuple{Vararg{AbstractUnitRange,N}} where {T, N} =
        new{T,N,Axes}(x,sz)
    Fill{T,0,Tuple{}}(x::T, sz::Tuple{}) where T = new{T,0,Tuple{}}(x,sz)
end
const FillVector{T} = Fill{T,1}
const FillMatrix{T} = Fill{T,2}
const FillVecOrMat{T} = Union{FillVector{T},FillMatrix{T}}

Fill{T,N,Axes}(x, sz::Axes) where Axes<:Tuple{Vararg{AbstractUnitRange,N}} where {T, N} =
    Fill{T,N,Axes}(convert(T, x)::T, sz)

Fill{T,0}(x, ::Tuple{}) where T = Fill{T,0,Tuple{}}(convert(T, x)::T, ()) # ambiguity fix

@inline Fill{T, N}(x, sz::Axes) where Axes<:Tuple{Vararg{AbstractUnitRange,N}} where {T, N} =
    Fill{T,N,Axes}(convert(T, x)::T, sz)

@inline Fill{T, N}(x, sz::SZ) where SZ<:Tuple{Vararg{Integer,N}} where {T, N} =
    Fill{T,N}(x, oneto.(sz))
@inline Fill{T, N}(x, sz::Vararg{Integer, N}) where {T, N} = Fill{T,N}(convert(T, x)::T, sz)


@inline Fill{T}(x, sz::Vararg{Integer,N}) where {T, N} = Fill{T, N}(x, sz)
@inline Fill{T}(x, sz::Tuple{Vararg{Any,N}}) where {T, N} = Fill{T, N}(x, sz)
""" `Fill(x, dims...)` construct lazy version of `fill(x, dims...)` """
@inline Fill(x::T, sz::Vararg{Integer,N}) where {T, N}  = Fill{T, N}(x, sz)
""" `Fill(x, dims)` construct lazy version of `fill(x, dims)` """
@inline Fill(x::T, sz::Tuple{Vararg{Any,N}}) where {T, N}  = Fill{T, N}(x, sz)

# We restrict to  when T is specified to avoid ambiguity with a Fill of a Fill
@inline Fill{T}(F::Fill{T}) where T = F
@inline Fill{T,N}(F::Fill{T,N}) where {T,N} = F
@inline Fill{T,N,Axes}(F::Fill{T,N,Axes}) where {T,N,Axes} = F

@inline axes(F::Fill) = F.axes
@inline size(F::Fill) = map(length, F.axes)

"""
    FillArrays.getindex_value(F::AbstractFill)

Return the value that `F` is filled with.

# Examples

```jldoctest
julia> f = Ones(3);

julia> FillArrays.getindex_value(f)
1.0

julia> g = Fill(2, 10);

julia> FillArrays.getindex_value(g)
2
```
"""
getindex_value

@inline getindex_value(F::Fill) = F.value

AbstractArray{T,N}(F::Fill{V,N}) where {T,V,N} = Fill{T}(convert(T, F.value)::T, F.axes)
AbstractFill{T}(F::AbstractFill) where T = AbstractArray{T}(F)
AbstractFill{T,N}(F::AbstractFill) where {T,N} = AbstractArray{T,N}(F)
AbstractFill{T,N,Ax}(F::AbstractFill{<:Any,N,Ax}) where {T,N,Ax} = AbstractArray{T,N}(F)

convert(::Type{AbstractFill{T}}, F::AbstractFill) where T = convert(AbstractArray{T}, F)
convert(::Type{AbstractFill{T,N}}, F::AbstractFill) where {T,N} = convert(AbstractArray{T,N}, F)
convert(::Type{AbstractFill{T,N,Ax}}, F::AbstractFill{<:Any,N,Ax}) where {T,N,Ax} = convert(AbstractArray{T,N}, F)

copy(F::Fill) = Fill(F.value, F.axes)

"""
    unique_value(arr::AbstractArray)

Return `only(unique(arr))` without intermediate allocations.
Throws an error if `arr` does not contain one and only one unique value.
"""
function unique_value(arr::AbstractArray)
    if isempty(arr) error("Cannot convert empty array to Fill") end
    val = first(arr)
    for x in arr
        if x !== val
            error("Input array contains both $x and $val. Cannot convert to Fill")
        end
    end
    return val
end
unique_value(f::AbstractFill) = getindex_value(f)
convert(::Type{Fill}, arr::AbstractArray{T}) where T = Fill{T}(unique_value(arr), axes(arr))
convert(::Type{Fill{T}}, arr::AbstractArray) where T = Fill{T}(unique_value(arr), axes(arr))
convert(::Type{Fill{T,N}}, arr::AbstractArray{<:Any,N}) where {T,N} = Fill{T,N}(unique_value(arr), axes(arr))
convert(::Type{Fill{T,N,Axes}}, arr::AbstractArray{<:Any,N}) where {T,N,Axes} = Fill{T,N,Axes}(unique_value(arr), axes(arr))
# ambiguity fix
convert(::Type{Fill}, arr::Fill{T}) where T = Fill{T}(unique_value(arr), axes(arr))
convert(::Type{T}, F::T) where T<:Fill = F



getindex(F::Fill{<:Any,0}) = getindex_value(F)

Base.@propagate_inbounds @inline function _fill_getindex(A::AbstractFill, I::Vararg{Union{Real, AbstractArray}, N}) where N
    @boundscheck checkbounds(A, I...)
    shape = Base.index_shape(I...)
    fillsimilar(A, shape)
end

Base.@propagate_inbounds @inline function _fill_getindex(A::AbstractFill, kr::AbstractArray{Bool})
   @boundscheck checkbounds(A, kr)
   fillsimilar(A, count(kr))
end

Base.@propagate_inbounds @inline Base._unsafe_getindex(::IndexStyle, F::AbstractFill, I::Vararg{Union{Real, AbstractArray}}) =
    _fill_getindex(F, I...)



Base.@propagate_inbounds getindex(A::AbstractFill, kr::AbstractVector{Bool}) = _fill_getindex(A, kr)
Base.@propagate_inbounds getindex(A::AbstractFill, kr::AbstractArray{Bool}) = _fill_getindex(A, kr)

@inline Base.iterate(F::AbstractFill) = length(F) == 0 ? nothing : (v = getindex_value(F); (v, (v, 1)))
@inline function Base.iterate(F::AbstractFill, (v, n))
    1 <= n < length(F) || return nothing
    v, (v, n+1)
end

# Iterators
Iterators.rest(F::AbstractFill, (_,n)) = fillsimilar(F, n <= 0 ? 0 : max(length(F)-n, 0))
function Iterators.drop(F::AbstractFill, n::Integer)
    n >= 0 || throw(ArgumentError("drop length must be nonnegative"))
    fillsimilar(F, max(length(F)-n, 0))
end
function Iterators.take(F::AbstractFill, n::Integer)
    n >= 0 || throw(ArgumentError("take length must be nonnegative"))
    fillsimilar(F, min(n, length(F)))
end
Base.rest(F::AbstractFill, s) = Iterators.rest(F, s)

#################
# Sorting
#################
function issorted(f::AbstractFill; kw...)
    v = getindex_value(f)
    issorted((v, v); kw...)
end
function sort(a::AbstractFill; kwds...)
    issorted(a; kwds...) # ensure that the values may be compared
    return a
end
function sort!(a::AbstractFill; kwds...)
    issorted(a; kwds...) # ensure that the values may be compared
    return a
end

svdvals!(a::AbstractFillMatrix) = [getindex_value(a)*sqrt(prod(size(a))); Zeros(min(size(a)...)-1)]

@noinline function _throw_dmrs(n, str, dims)
    throw(DimensionMismatch("parent has $n elements, which is incompatible with $str $dims"))
end
function fill_reshape(parent, dims::Integer...)
    n = length(parent)
    prod(dims) == n || _throw_dmrs(n, "size", dims)
    fillsimilar(parent, dims...)
end

reshape(parent::AbstractFill, dims::Integer...) = reshape(parent, dims)
reshape(parent::AbstractFill, dims::Union{Int,Colon}...) = reshape(parent, dims)
reshape(parent::AbstractFill, dims::Union{Integer,Colon}...) = reshape(parent, dims)
# resolve ambiguity with Base
reshape(parent::AbstractFillVector, dims::Colon) = parent

reshape(parent::AbstractFill, dims::Tuple{Vararg{Union{Integer,Colon}}}) =
    fill_reshape(parent, Base._reshape_uncolon(parent, dims)...)
reshape(parent::AbstractFill, dims::Tuple{Vararg{Union{Int,Colon}}}) =
    fill_reshape(parent, Base._reshape_uncolon(parent, dims)...)
reshape(parent::AbstractFill, shp::Tuple{Union{Integer,Base.OneTo}, Vararg{Union{Integer,Base.OneTo}}}) =
    reshape(parent, Base.to_shape(shp))
reshape(parent::AbstractFill, dims::Dims) = fill_reshape(parent, dims...)
reshape(parent::AbstractFill, dims::Tuple{Integer, Vararg{Integer}}) = fill_reshape(parent, dims...)

# resolve ambiguity with Base
reshape(parent::AbstractFillVector, dims::Tuple{Colon}) = parent

for (AbsTyp, Typ, funcs, func) in ((:AbstractZeros, :Zeros, :zeros, :zero), (:AbstractOnes, :Ones, :ones, :one))
    @eval begin
        abstract type $AbsTyp{T, N, Axes} <: AbstractFill{T, N, Axes} end
        $(Symbol(AbsTyp,"Vector")){T} = $AbsTyp{T,1}
        $(Symbol(AbsTyp,"Matrix")){T} = $AbsTyp{T,2}
        $(Symbol(AbsTyp,"VecOrMat")){T} = Union{$(Symbol(AbsTyp,"Vector")){T},$(Symbol(AbsTyp,"Matrix"))}

        """ `$($Typ){T, N, Axes} <: AbstractFill{T, N, Axes}` (lazy `$($funcs)` with axes)"""
        struct $Typ{T, N, Axes} <: $AbsTyp{T, N, Axes}
            axes::Axes
            @inline $Typ{T,N,Axes}(sz::Axes) where Axes<:Tuple{Vararg{AbstractUnitRange,N}} where {T, N} =
                new{T,N,Axes}(sz)
            @inline $Typ{T,N}(sz::Axes) where Axes<:Tuple{Vararg{AbstractUnitRange,N}} where {T, N} =
                new{T,N,Axes}(sz)
            @inline $Typ{T,0,Tuple{}}(sz::Tuple{}) where T = new{T,0,Tuple{}}(sz)
        end
        const $(Symbol(Typ,"Vector")){T} = $Typ{T,1}
        const $(Symbol(Typ,"Matrix")){T} = $Typ{T,2}
        const $(Symbol(Typ,"VecOrMat")){T} = Union{$Typ{T,1},$Typ{T,2}}


        @inline $Typ{T, 0}(sz::Tuple{}) where {T} = $Typ{T,0,Tuple{}}(sz)
        @inline $Typ{T, N}(sz::Tuple{Vararg{Integer, N}}) where {T, N} = $Typ{T,N}(oneto.(sz))
        @inline $Typ{T, N}(sz::Vararg{Integer, N}) where {T, N} = $Typ{T,N}(sz)
        """ `$($Typ){T}(dims...)` construct lazy version of `$($funcs)(dims...)`"""
        @inline $Typ{T}(sz::Vararg{Integer,N}) where {T, N} = $Typ{T, N}(sz)
        @inline $Typ{T}(sz::SZ) where SZ<:Tuple{Vararg{Any,N}} where {T, N} = $Typ{T, N}(sz)
        @inline $Typ(sz::Vararg{Any,N}) where N = $Typ{Float64,N}(sz)
        @inline $Typ(sz::SZ) where SZ<:Tuple{Vararg{Any,N}} where N = $Typ{Float64,N}(sz)
        @inline $Typ{T}(n::Integer) where T = $Typ{T,1}(n)
        @inline $Typ(n::Integer) = $Typ{Float64,1}(n)

        @inline $Typ{T,N,Axes}(A::AbstractArray{V,N}) where{T,V,N,Axes} = $Typ{T,N,Axes}(axes(A))
        @inline $Typ{T,N}(A::AbstractArray{V,N}) where{T,V,N} = $Typ{T,N}(size(A))
        @inline $Typ{T}(A::AbstractArray) where{T} = $Typ{T}(size(A))
        @inline $Typ(A::AbstractArray) = $Typ{eltype(A)}(A)
        @inline $Typ(::Type{T}, m...) where T = $Typ{T}(m...)

        @inline axes(Z::$Typ) = Z.axes
        @inline size(Z::$AbsTyp) = length.(axes(Z))
        @inline getindex_value(Z::$AbsTyp{T}) where T = $func(T)

        AbstractArray{T}(F::$AbsTyp) where T = $Typ{T}(axes(F))
        AbstractArray{T,N}(F::$AbsTyp{V,N}) where {T,V,N} = $Typ{T}(axes(F))

        copy(F::$AbsTyp) = F

        getindex(F::$AbsTyp{T,0}) where T = getindex_value(F)

        promote_rule(::Type{$Typ{T, N, Axes}}, ::Type{$Typ{V, N, Axes}}) where {T,V,N,Axes} = $Typ{promote_type(T,V),N,Axes}
        function convert(::Type{Typ}, A::$AbsTyp{V,N,Axes}) where {T,V,N,Axes,Typ<:$AbsTyp{T,N,Axes}}
            convert(T, getindex_value(A)) # checks that the types are convertible
            Typ(axes(A))
        end
        convert(::Type{$Typ{T,N}}, A::$AbsTyp{V,N,Axes}) where {T,V,N,Axes} = convert($Typ{T,N,Axes}, A)
        convert(::Type{$Typ{T}}, A::$AbsTyp{V,N,Axes}) where {T,V,N,Axes} = convert($Typ{T,N,Axes}, A)
        function convert(::Type{Typ}, A::AbstractFill{V,N}) where {T,V,N,Axes,Typ<:$AbsTyp{T,N,Axes}}
            axes(A) isa Axes || throw(ArgumentError("cannot convert, as axes of array are not $Axes"))
            val = getindex_value(A)
            y = convert(T, val)
            y == $func(T) || throw(ArgumentError(string("cannot convert an array containinig $val to ", $Typ)))
            Typ(axes(A))
        end
        function convert(::Type{$Typ{T,N}}, A::AbstractFill{<:Any,N}) where {T,N}
            convert($Typ{T,N,typeof(axes(A))}, A)
        end
        function convert(::Type{$Typ{T}}, A::AbstractFill{<:Any,N}) where {T,N}
            convert($Typ{T,N}, A)
        end
        function convert(::Type{$Typ}, A::AbstractFill{V,N}) where {V,N}
            convert($Typ{V,N}, A)
        end
    end
end

# conversions
for TYPE in (:Fill, :AbstractFill, :Ones, :Zeros), STYPE in (:AbstractArray, :AbstractFill)
    @eval begin
        @inline $STYPE{T}(F::$TYPE{T}) where T = F
        @inline $STYPE{T,N}(F::$TYPE{T,N}) where {T,N} = F
    end
end

promote_rule(::Type{<:AbstractFill{T, N, Axes}}, ::Type{<:AbstractFill{V, N, Axes}}) where {T,V,N,Axes} = Fill{promote_type(T,V),N,Axes}

"""
    fillsimilar(a::AbstractFill, axes...)

creates a fill object that has the same fill value as `a` but
with the specified axes.
For example, if `a isa Zeros` then so is the returned object.
"""
fillsimilar(a::Ones{T}, axes...) where T = Ones{T}(axes...)
fillsimilar(a::Zeros{T}, axes...) where T = Zeros{T}(axes...)
fillsimilar(a::AbstractFill, axes...) = Fill(getindex_value(a), axes...)

# functions
function Base.sqrt(a::AbstractFillMatrix{<:Union{Real, Complex}})
    Base.require_one_based_indexing(a)
    size(a,1) == size(a,2) || throw(DimensionMismatch("matrix is not square: dimensions are $(size(a))"))
    _sqrt(a)
end
_sqrt(a::AbstractZerosMatrix) = float(a)
function _sqrt(a::AbstractFillMatrix)
    n = size(a,1)
    n == 0 && return float(a)
    v = getindex_value(a)
    Fill(√(v/n), axes(a))
end
function Base.cbrt(a::AbstractFillMatrix{<:Real})
    Base.require_one_based_indexing(a)
    size(a,1) == size(a,2) || throw(DimensionMismatch("matrix is not square: dimensions are $(size(a))"))
    _cbrt(a)
end
_cbrt(a::AbstractZerosMatrix) = float(a)
function _cbrt(a::AbstractFillMatrix)
    n = size(a,1)
    n == 0 && return float(a)
    v = getindex_value(a)
    Fill(cbrt(v)/cbrt(n)^2, axes(a))
end

struct RectDiagonal{T,V<:AbstractVector{T},Axes<:Tuple{Vararg{AbstractUnitRange,2}}} <: AbstractMatrix{T}
    diag::V
    axes::Axes

    @inline function RectDiagonal{T,V}(A::V, axes::Axes) where {T,V<:AbstractVector{T},Axes<:Tuple{Vararg{AbstractUnitRange,2}}}
        Base.require_one_based_indexing(A)
        @assert any(length(ax) == length(A) for ax in axes)
        rd = new{T,V,Axes}(A, axes)
        Base.require_one_based_indexing(rd)
        return rd
    end
end

@inline RectDiagonal{T,V}(A::V, sz::Tuple{Vararg{Integer, 2}}) where {T,V} = RectDiagonal{T,V}(A, oneto.(sz))
@inline RectDiagonal{T,V}(A::V, axes::Vararg{Any, 2}) where {T,V} = RectDiagonal{T,V}(A, axes)
@inline RectDiagonal{T,V}(A::V, sz::Vararg{Integer, 2}) where {T,V} = RectDiagonal{T,V}(A, sz)
@inline RectDiagonal{T,V}(A::V) where {T,V} = RectDiagonal{T,V}(A, (axes(A, 1), axes(A, 1)))
@inline RectDiagonal{T}(A::V, args...) where {T,V} = RectDiagonal{T,V}(A, args...)
@inline RectDiagonal(A::V, args...) where {V} = RectDiagonal{eltype(V),V}(A, args...)

const UpperOrUnitUpperTriangular{T,S} = Union{UpperTriangular{T,S}, UnitUpperTriangular{T,S}}
const LowerOrUnitLowerTriangular{T,S} = Union{LowerTriangular{T,S}, UnitLowerTriangular{T,S}}
const UpperOrLowerTriangular{T,S} = Union{UpperOrUnitUpperTriangular{T,S}, LowerOrUnitLowerTriangular{T,S}}

# patch missing overload from Base
axes(rd::Diagonal{<:Any,<:AbstractFill}) = (axes(rd.diag,1),axes(rd.diag,1))
axes(T::UpperOrLowerTriangular{<:Any,<:AbstractFill}) = axes(parent(T))

axes(rd::RectDiagonal) = rd.axes
size(rd::RectDiagonal) = map(length, rd.axes)

parent(rd::RectDiagonal) = rd.diag

@inline function getindex(rd::RectDiagonal{T}, i::Integer, j::Integer) where T
    @boundscheck checkbounds(rd, i, j)
    if i == j
        @inbounds r = rd.diag[i]
    else
        r = zero(T)
    end
    return r
end

function setindex!(rd::RectDiagonal, v, i::Integer, j::Integer)
    @boundscheck checkbounds(rd, i, j)
    if i == j
        @inbounds rd.diag[i] = v
    elseif !iszero(v)
        throw(ArgumentError("cannot set off-diagonal entry ($i, $j) to a nonzero value ($v)"))
    end
    return v
end

diag(rd::RectDiagonal) = rd.diag

for f in (:triu, :triu!, :tril, :tril!)
    @eval ($f)(M::RectDiagonal) = M
end

# Due to default definitions in LinearAlgebra only the following implementations are needed
# (see above for more details)
function +(a::RectDiagonal, b::UniformScaling)
    LinearAlgebra.checksquare(a)
    return Diagonal(a.diag .+ b.λ)
end
function -(a::UniformScaling, b::RectDiagonal)
    LinearAlgebra.checksquare(b)
    return Diagonal(a.λ .- b.diag)
end

Base.replace_in_print_matrix(A::RectDiagonal, i::Integer, j::Integer, s::AbstractString) =
    i == j ? s : Base.replace_with_centered_mark(s)


const RectOrDiagonal{T,V,Axes} = Union{RectDiagonal{T,V,Axes}, Diagonal{T,V}}
const RectOrDiagonalFill{T,V<:AbstractFillVector{T},Axes} = RectOrDiagonal{T,V,Axes}
const RectDiagonalFill{T,V<:AbstractFillVector{T}} = RectDiagonal{T,V}
const SquareEye{T,Axes} = Diagonal{T,Ones{T,1,Tuple{Axes}}}
const Eye{T,Axes} = RectOrDiagonal{T,Ones{T,1,Tuple{Axes}}}

@inline SquareEye{T}(n::Integer) where T = Diagonal(Ones{T}(n))
@inline SquareEye(n::Integer) = Diagonal(Ones(n))
@inline SquareEye{T}(ax::Tuple{AbstractUnitRange{Int}}) where T = Diagonal(Ones{T}(ax))
@inline SquareEye(ax::Tuple{AbstractUnitRange{Int}}) = Diagonal(Ones(ax))

@inline Eye{T}(n::Integer) where T = SquareEye{T}(n)
@inline Eye(n::Integer) = SquareEye(n)
@inline Eye{T}(ax::Tuple{AbstractUnitRange{Int}}) where T = SquareEye{T}(ax)
@inline Eye(ax::Tuple{AbstractUnitRange{Int}}) = SquareEye(ax)

# function iterate(iter::Eye, istate = (1, 1))
#     (i::Int, j::Int) = istate
#     m = size(iter, 1)
#     return i > m ? nothing :
#         ((@inbounds getindex(iter, i, j)),
#          j == m ? (i + 1, 1) : (i, j + 1))
# end

isone(::SquareEye) = true

function diag(E::Eye, k::Integer=0)
    v = k == 0 ? oneunit(eltype(E)) : zero(eltype(E))
    len = length(diagind(E, k))
    Fill(v, len)
end

# These should actually be in StdLib, LinearAlgebra.jl, for all Diagonal
for f in (:permutedims, :triu, :triu!, :tril, :tril!, :copy)
    @eval ($f)(IM::Diagonal{<:Any,<:AbstractFill}) = IM
end

inv(IM::SquareEye) = IM
inv(IM::Diagonal{<:Any,<:AbstractFill}) = Diagonal(map(inv, IM.diag))

Eye(n::Integer, m::Integer) = RectDiagonal(Ones(min(n,m)), n, m)
Eye{T}(n::Integer, m::Integer) where T = RectDiagonal{T}(Ones{T}(min(n,m)), n, m)
function Eye{T}((a,b)::NTuple{2,AbstractUnitRange{Int}}) where T
    ab = length(a) ≤ length(b) ? a : b
    RectDiagonal{T}(Ones{T}((ab,)), (a,b))
end
function Eye((a,b)::NTuple{2,AbstractUnitRange{Int}})
    ab = length(a) ≤ length(b) ? a : b
    RectDiagonal(Ones((ab,)), (a,b))
end

@inline Eye{T}(A::AbstractMatrix) where T = Eye{T}(size(A)...)
@inline Eye(A::AbstractMatrix) = Eye{eltype(A)}(size(A)...)

# This may break, as it uses undocumented internals of LinearAlgebra
# Ideally this should be copied over to this package
# Also, maybe this should reuse the broadcasting behavior of the parent,
# once AbstractFill types implement their own BroadcastStyle
BroadcastStyle(::Type{<:RectDiagonal}) = LinearAlgebra.StructuredMatrixStyle{RectDiagonal}()
LinearAlgebra.structured_broadcast_alloc(bc, ::Type{<:RectDiagonal}, ::Type{ElType}, n) where {ElType} =
    RectDiagonal(Array{ElType}(undef, n), axes(bc))
@inline LinearAlgebra.fzero(S::RectDiagonal{T}) where {T} = zero(T)

#########
#  Special matrix types
#########



## Array
Base.Array{T,N}(F::AbstractFill{V,N}) where {T,V,N} =
    convert(Array{T,N}, fill(convert(T, getindex_value(F)), size(F)))

# These are in case `zeros` or `ones` are ever faster than `fill`
for (Typ, funcs, func) in ((:AbstractZeros, :zeros, :zero), (:AbstractOnes, :ones, :one))
    @eval begin
        Base.Array{T,N}(F::$Typ{V,N}) where {T,V,N} = $funcs(T,size(F))
    end
end

if VERSION < v"1.11-"
    # temporary patch. should be a PR(#48895) to LinearAlgebra
    Diagonal{T}(A::AbstractFillMatrix) where T = Diagonal{T}(diag(A))
    function convert(::Type{T}, A::AbstractFillMatrix) where T<:Diagonal
        checksquare(A)
        isdiag(A) ? T(diag(A)) : throw(InexactError(:convert, T, A))
    end
end

Base.StepRangeLen(F::AbstractFillVector{T}) where T = StepRangeLen(getindex_value(F), zero(T), length(F))
convert(::Type{SL}, F::AbstractFillVector) where SL<:AbstractRange = convert(SL, StepRangeLen(F))

#################
# Structured matrix types
#################

for SMT in (:Diagonal, :Bidiagonal, :Tridiagonal, :SymTridiagonal)
    @eval function diag(D::$SMT{T,<:AbstractFillVector{T}}, k::Integer=0) where {T<:Number}
        inds = (1,1) .+ (k >= 0 ? (0,k) : (-k,0))
        v = get(D, inds, zero(eltype(D)))
        Fill(v, length(diagind(D, k)))
    end
end


#########
# maximum/minimum
#########

for op in (:maximum, :minimum)
    @eval $op(x::AbstractFill) = getindex_value(x)
end


#########
# Cumsum
#########

# These methods are necessary to deal with infinite arrays
sum(x::AbstractFill) = getindex_value(x)*length(x)
sum(f, x::AbstractFill) = length(x) * f(getindex_value(x))
sum(x::AbstractZeros) = getindex_value(x)

# needed to support infinite case
steprangelen(st...) = StepRangeLen(st...)
function cumsum(x::AbstractFill{T,1}) where T
    V = promote_op(add_sum, T, T)
    steprangelen(convert(V,getindex_value(x)), getindex_value(x), length(x))
end

cumsum(x::AbstractZerosVector{T}) where T = _range_convert(AbstractVector{promote_op(add_sum, T, T)}, x)
cumsum(x::AbstractZerosVector{Bool}) = _range_convert(AbstractVector{Int}, x)
cumsum(x::AbstractOnesVector{T}) where T<:Integer = _range_convert(AbstractVector{promote_op(add_sum, T, T)}, oneto(length(x)))
cumsum(x::AbstractOnesVector{Bool}) = oneto(length(x))


for op in (:+, :-)
    @eval begin
        function accumulate(::typeof($op), x::AbstractFill{T,1}) where T
            V = promote_op($op, T, T)
            steprangelen(convert(V,getindex_value(x)), $op(getindex_value(x)), length(x))
        end

        accumulate(::typeof($op), x::AbstractZerosVector{T}) where T = _range_convert(AbstractVector{promote_op($op, T, T)}, x)
        accumulate(::typeof($op), x::AbstractZerosVector{Bool}) = _range_convert(AbstractVector{Int}, x)
    end
end

accumulate(::typeof(+), x::AbstractOnesVector{T}) where T<:Integer = _range_convert(AbstractVector{promote_op(+, T, T)}, oneto(length(x)))
accumulate(::typeof(+), x::AbstractOnesVector{Bool}) = oneto(length(x))

#########
# Diff
#########

diff(x::AbstractFillVector{T}) where T = Zeros{T}(length(x)-1)

#########
# unique
#########

unique(x::AbstractFill) = fillsimilar(x, Int(!isempty(x)))
allunique(x::AbstractFill) = length(x) < 2

#########
# zero
#########

zero(r::AbstractZeros{T,N}) where {T,N} = r
# TODO: Make this required?
zero(r::AbstractOnes{T,N}) where {T,N} = Zeros{T,N}(axes(r))
zero(r::Fill{T,N}) where {T,N} = Zeros{T,N}(r.axes)

#########
# oneunit
#########

function one(A::AbstractFillMatrix{T}) where {T}
    Base.require_one_based_indexing(A)
    m, n = size(A)
    m == n || throw(ArgumentError("multiplicative identity defined only for square matrices"))
    SquareEye{T}(m)
end

#########
# any/all/isone/iszero
#########

function isone(AF::AbstractFillMatrix)
    (n,m) = size(AF)
    n != m && return false
    (n == 0 || m == 0) && return true
    isone(getindex_value(AF)) || return false
    n == 1 && return true
    return false
end

# all(isempty, []) and any(isempty, []) have non-generic behavior.
# We do not follow it here for Eye(0).
function any(f::Function, IM::Eye{T}) where T
    d1, d2 = size(IM)
    (d1 < 1 || d2 < 1) && return false
    (d1 > 1 || d2 > 1) && return f(zero(T)) || f(one(T))
    return any(f(one(T)))
end

function all(f::Function, IM::Eye{T}) where T
    d1, d2 = size(IM)
    (d1 < 1 || d2 < 1) && return false
    (d1 > 1 || d2 > 1) && return f(zero(T)) && f(one(T))
    return all(f(one(T)))
end

# In particular, these make iszero(Eye(n))  efficient.
# use any/all on scalar to get Boolean error message
function any(f::Function, x::AbstractFill)
    isempty(x) && return false
    # If the condition is true for one value, then it's true for all
    fval = f(getindex_value(x))
    any((fval,))
end
function all(f::Function, x::AbstractFill)
    isempty(x) && return true
    # If the condition is true for one value, then it's true for all
    fval = f(getindex_value(x))
    return all((fval,))
end
any(x::AbstractFill) = any(identity, x)
all(x::AbstractFill) = all(identity, x)

count(x::AbstractOnes{Bool}) = length(x)
count(x::AbstractZeros{Bool}) = 0
count(f, x::AbstractFill) = f(getindex_value(x)) ? length(x) : 0

#########
# in
#########
in(x, A::AbstractFill) = x == getindex_value(A)
function in(x, A::RectDiagonal{<:Number})
    any(iszero, size(A)) && return false # Empty matrix
    all(isone, size(A)) && return x == A.diag[1] # A 1x1 matrix has only one element
    x == zero(eltype(A)) || x in A.diag
end

#########
# include
#########

include("fillalgebra.jl")
include("fillbroadcast.jl")
include("trues.jl")

##
# print
##
Base.replace_in_print_matrix(::AbstractZeros, ::Integer, ::Integer, s::AbstractString) =
    Base.replace_with_centered_mark(s)

# following support blocked fill array printing via
# BlockArrays.jl
axes_print_matrix_row(lay, io, X, A, i, cols, sep, idxlast::Integer=last(axes(X, 2))) =
    Base.invoke(Base.print_matrix_row, Tuple{IO,AbstractVecOrMat,Vector,Integer,AbstractVector,AbstractString,Integer},
                    io, X, A, i, cols, sep, idxlast)

Base.print_matrix_row(io::IO,
        X::Union{AbstractFillVector,
                 AbstractFillMatrix,
                 Diagonal{<:Any,<:AbstractFillVector},
                 RectDiagonal,
                 UpperOrLowerTriangular{<:Any,<:AbstractFillMatrix}
                 }, A::Vector,
        i::Integer, cols::AbstractVector, sep::AbstractString, idxlast::Integer=last(axes(X, 2))) =
        axes_print_matrix_row(axes(X), io, X, A, i, cols, sep)


# Display concise description of a Fill.

function Base.show(io::IO, ::MIME"text/plain", x::Union{Eye, AbstractFill})
    if get(IOContext(io), :compact, false)  # for example [Fill(i==j,2,2) for i in 1:3, j in 1:4]
        return show(io, x)
    end
    summary(io, x)
    if !(x isa Union{AbstractZeros, AbstractOnes, Eye})
        print(io, ", with ", length(x) > 1 ? "entries" : "entry", " equal to ")
        show(io, getindex_value(x))
    end
end

function Base.show(io::IO, x::AbstractFill)  # for example (Fill(π,3),)
    print(io, nameof(typeof(x)))
    sz = size(x)
    args = if x isa Union{AbstractZeros, AbstractOnes}
        T = eltype(x)
        if T != Float64
            print(io,"{", T, "}")
        end
        print(io, "(")
    else
        # show, not print, to handle (Fill(1f0,2),)
        print(io, "(")
        show(io, getindex_value(x))
        ndims(x) == 0 || print(io, ", ")
    end
    join(io, size(x), ", ")
    print(io, ")")
end
function Base.show(io::IO, x::Eye)
    print(io, "Eye(", size(x,1))
    if size(x,1) != size(x,2)
        print(io, ",", size(x,2))
    end
    print(io, ")")
end

Base.array_summary(io::IO, ::Zeros{T}, inds::Tuple{Vararg{Base.OneTo}}) where T =
    print(io, Base.dims2string(length.(inds)), " Zeros{$T}")
Base.array_summary(io::IO, ::Ones{T}, inds::Tuple{Vararg{Base.OneTo}}) where T =
    print(io, Base.dims2string(length.(inds)), " Ones{$T}")
Base.array_summary(io::IO, a::Fill{T}, inds::Tuple{Vararg{Base.OneTo}}) where T =
    print(io, Base.dims2string(length.(inds)), " Fill{$T}")
Base.array_summary(io::IO, a::Eye{T}, inds::Tuple{Vararg{Base.OneTo}}) where T =
    print(io, Base.dims2string(length.(inds)), " Eye{$T}")


##
# interface
##

getindex_value(a::LinearAlgebra.Adjoint) = adjoint(getindex_value(parent(a)))
getindex_value(a::LinearAlgebra.Transpose) = transpose(getindex_value(parent(a)))
getindex_value(a::SubArray) = getindex_value(parent(a))

copy(a::LinearAlgebra.Adjoint{<:Any,<:AbstractFill}) = copy(parent(a))'
copy(a::LinearAlgebra.Transpose{<:Any,<:AbstractFill}) = transpose(parent(a))

##
# view
##

Base.@propagate_inbounds view(A::AbstractFill{<:Any,N}, kr::AbstractArray{Bool,N}) where N = _fill_getindex(A, kr)
Base.@propagate_inbounds view(A::AbstractFill{<:Any,1}, kr::AbstractVector{Bool}) = _fill_getindex(A, kr)
Base.@propagate_inbounds view(A::AbstractFill, I...) =
    _fill_getindex(A, Base.to_indices(A,I)...)

# not getindex since we need array-like indexing
Base.@propagate_inbounds function view(A::AbstractFill, I::Vararg{Real})
    @boundscheck checkbounds(A, I...)
    fillsimilar(A)
end

# repeat

_first(t::Tuple) = t[1]
_first(t::Tuple{}) = 1

_maybetail(t::Tuple) = Base.tail(t)
_maybetail(t::Tuple{}) = t

_match_size(sz::Tuple{}, inner::Tuple{}, outer::Tuple{}) = ()
function _match_size(sz::Tuple, inner::Tuple, outer::Tuple)
    t1 = (_first(sz), _first(inner), _first(outer))
    t2 = _match_size(_maybetail(sz), _maybetail(inner), _maybetail(outer))
    (t1, t2...)
end

function _repeat_size(sz::Tuple, inner::Tuple, outer::Tuple)
    t = _match_size(sz, inner, outer)
    map(*, getindex.(t, 1), getindex.(t, 2), getindex.(t, 3))
end

function _repeat(A; inner=ntuple(x->1, ndims(A)), outer=ntuple(x->1, ndims(A)))
    Base.require_one_based_indexing(A)
    length(inner) >= ndims(A) ||
        throw(ArgumentError("number of inner repetitions $(length(inner)) cannot be "*
            "less than number of dimensions of input array $(ndims(A))"))
    length(outer) >= ndims(A) ||
        throw(ArgumentError("number of outer repetitions $(length(outer)) cannot be "*
            "less than number of dimensions of input array $(ndims(A))"))
    sz = _repeat_size(size(A), Tuple(inner), Tuple(outer))
    fillsimilar(A, sz)
end

repeat(A::AbstractFill, count::Integer...) = _repeat(A, outer=count)
function repeat(A::AbstractFill; inner=ntuple(x->1, ndims(A)), outer=ntuple(x->1, ndims(A)))
    _repeat(A, inner=inner, outer=outer)
end

include("oneelement.jl")

end # module
