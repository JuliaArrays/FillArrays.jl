""" `FillArrays` module to lazily represent matrices with a single value """
module FillArrays

using LinearAlgebra, SparseArrays
import Base: size, getindex, setindex!, IndexStyle, checkbounds, convert,
    +, -, *, /, \, diff, sum, cumsum, maximum, minimum, sort, sort!,
    any, all, axes, isone, iterate, unique, allunique, permutedims, inv,
    copy, vec, setindex!, count, ==, reshape, _throw_dmrs, map, zero,
    show

import LinearAlgebra: rank, svdvals!, tril, triu, tril!, triu!, diag, transpose, adjoint, fill!,
    norm2, norm1, normInf, normMinusInf, normp, lmul!, rmul!, diagzero, AbstractTriangular

import Base.Broadcast: broadcasted, DefaultArrayStyle, broadcast_shape



export Zeros, Ones, Fill, Eye

"""
    AbstractFill{T, N, Axes} <: AbstractArray{T, N}

Supertype for lazy array types whose entries are all equal to constant.
"""
abstract type AbstractFill{T, N, Axes} <: AbstractArray{T, N} end

==(a::AbstractFill, b::AbstractFill) = axes(a) == axes(b) && getindex_value(a) == getindex_value(b)

@inline function getindex(F::AbstractFill, k::Integer)
    @boundscheck checkbounds(F, k)
    getindex_value(F)
end

@inline function getindex(F::AbstractFill{T, N}, kj::Vararg{<:Integer, N}) where {T, N}
    @boundscheck checkbounds(F, kj...)
    getindex_value(F)
end

@inline function setindex!(F::AbstractFill, v, k::Integer)
    @boundscheck checkbounds(F, k)
    v == getindex_value(F) || throw(ArgumentError("Cannot setindex! to $v for an AbstractFill with value $(getindex_value(F))."))
    F
end

@inline function setindex!(F::AbstractFill{T, N}, v, kj::Vararg{<:Integer, N}) where {T, N}
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


"""
    Fill{T, N, Axes}
    where `Axes <: Tuple{Vararg{AbstractUnitRange,N}}`

A lazy representation of an array of dimension `N`
whose entries are all equal to a constant of type `T`,
with axes of type `Axes`.
Typically created by `Fill` or `Zeros` or `Ones`

# Examples

```jldoctest
julia> Fill(7, (2,3))
2×3 Fill{Int64,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}:
 7  7  7
 7  7  7

julia> Fill{Float64, 1, Tuple{UnitRange{Int64}}}(7., (1:2,))
2-element Fill{Float64,1,Tuple{UnitRange{Int64}}} with indices 1:2:
 7.0
 7.0
```
"""
struct Fill{T, N, Axes} <: AbstractFill{T, N, Axes}
    value::T
    axes::Axes

    Fill{T,N,Axes}(x::T, sz::Axes) where Axes<:Tuple{Vararg{AbstractUnitRange,N}} where {T, N} =
        new{T,N,Axes}(x,sz)
    Fill{T,0,Tuple{}}(x::T, sz::Tuple{}) where T = new{T,0,Tuple{}}(x,sz)
end

Fill{T,0}(x::T, ::Tuple{}) where T = Fill{T,0,Tuple{}}(x, ()) # ambiguity fix

@inline Fill{T, N}(x::T, sz::Axes) where Axes<:Tuple{Vararg{AbstractUnitRange,N}} where {T, N} =
    Fill{T,N,Axes}(x, sz)
@inline Fill{T, N}(x, sz::Axes) where Axes<:Tuple{Vararg{AbstractUnitRange,N}} where {T, N} =
    Fill{T,N}(convert(T, x)::T, sz)

@inline Fill{T, N}(x, sz::SZ) where SZ<:Tuple{Vararg{Integer,N}} where {T, N} =
    Fill{T,N}(x, Base.OneTo.(sz))
@inline Fill{T, N}(x, sz::Vararg{Integer, N}) where {T, N} = Fill{T,N}(convert(T, x)::T, sz)


@inline Fill{T}(x, sz::Vararg{<:Integer,N}) where {T, N} = Fill{T, N}(x, sz)
@inline Fill{T}(x, sz::Tuple{Vararg{<:Any,N}}) where {T, N} = Fill{T, N}(x, sz)
""" `Fill(x, dims...)` construct lazy version of `fill(x, dims...)` """
@inline Fill(x::T, sz::Vararg{<:Integer,N}) where {T, N}  = Fill{T, N}(x, sz)
""" `Fill(x, dims)` construct lazy version of `fill(x, dims)` """
@inline Fill(x::T, sz::Tuple{Vararg{<:Any,N}}) where {T, N}  = Fill{T, N}(x, sz)

# We restrict to  when T is specified to avoid ambiguity with a Fill of a Fill
@inline Fill{T}(F::Fill{T}) where T = F
@inline Fill{T,N}(F::Fill{T,N}) where {T,N} = F
@inline Fill{T,N,Axes}(F::Fill{T,N,Axes}) where {T,N,Axes} = F

@inline axes(F::Fill) = F.axes
@inline size(F::Fill) = length.(F.axes)

@inline getindex_value(F::Fill) = F.value

AbstractArray{T}(F::Fill{T}) where T = F
AbstractArray{T,N}(F::Fill{T,N}) where {T,N} = F
AbstractArray{T}(F::Fill{V,N}) where {T,V,N} = Fill{T}(convert(T, F.value)::T, F.axes)
AbstractArray{T,N}(F::Fill{V,N}) where {T,V,N} = Fill{T}(convert(T, F.value)::T, F.axes)

convert(::Type{AbstractArray{T}}, F::Fill{T}) where T = F
convert(::Type{AbstractArray{T,N}}, F::Fill{T,N}) where {T,N} = F
convert(::Type{AbstractArray{T}}, F::Fill) where {T} = AbstractArray{T}(F)
convert(::Type{AbstractArray{T,N}}, F::Fill) where {T,N} = AbstractArray{T,N}(F)
convert(::Type{AbstractFill}, F::AbstractFill) = F
convert(::Type{AbstractFill{T}}, F::AbstractFill) where T = convert(AbstractArray{T}, F)
convert(::Type{AbstractFill{T,N}}, F::AbstractFill) where {T,N} = convert(AbstractArray{T,N}, F)

copy(F::Fill) = Fill(F.value, F.axes)

""" Throws an error if `arr` does not contain one and only one unique value. """
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
convert(::Type{T}, F::T) where T<:Fill = F   # ambiguity fix



getindex(F::Fill{<:Any,0}) = getindex_value(F)

function getindex(F::Fill, kj::Vararg{AbstractVector{II},N}) where {II<:Integer,N}
    checkbounds(F, kj...)
    Fill(getindex_value(F), length.(kj))
end

function getindex(A::Fill, kr::AbstractVector{Bool})
   length(A) == length(kr) || throw(DimensionMismatch())
   Fill(getindex_value(A), count(kr))
end
function getindex(A::Fill, kr::AbstractArray{Bool})
   size(A) == size(kr) || throw(DimensionMismatch())
   Fill(getindex_value(A), count(kr))
end

sort(a::AbstractFill; kwds...) = a
sort!(a::AbstractFill; kwds...) = a
svdvals!(a::AbstractFill{<:Any,2}) = [getindex_value(a)*sqrt(prod(size(a))); Zeros(min(size(a)...)-1)]

+(a::AbstractFill) = a
-(a::AbstractFill) = Fill(-getindex_value(a), size(a))

# Fill +/- Fill
function +(a::AbstractFill{T, N}, b::AbstractFill{V, N}) where {T, V, N}
    axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
    return Fill(getindex_value(a) + getindex_value(b), axes(a))
end
-(a::AbstractFill, b::AbstractFill) = a + (-b)

function +(a::Fill{T, 1}, b::AbstractRange) where {T}
    size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
    Tout = promote_type(T, eltype(b))
    return (a.value + first(b)):convert(Tout, step(b)):(a.value + last(b))
end
function +(a::Fill{T, 1}, b::UnitRange) where {T}
    size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
    Tout = promote_type(T, eltype(b))
    return (a.value + first(b)):(a.value + last(b))
end
+(a::AbstractRange, b::AbstractFill) = b + a

-(a::AbstractFill, b::AbstractRange) = a + (-b)
-(a::AbstractRange, b::AbstractFill) = a + (-b)

function fill_reshape(parent, dims::Integer...)
    n = length(parent)
    prod(dims) == n || _throw_dmrs(n, "size", dims)
    Fill(getindex_value(parent), dims...)
end

reshape(parent::AbstractFill, dims::Integer...) = reshape(parent, dims)
reshape(parent::AbstractFill, dims::Union{Int,Colon}...) = reshape(parent, dims)
reshape(parent::AbstractFill, dims::Union{Integer,Colon}...) = reshape(parent, dims)

reshape(parent::AbstractFill, dims::Tuple{Vararg{Union{Integer,Colon}}}) = 
    fill_reshape(parent, Base._reshape_uncolon(parent, dims)...)
reshape(parent::AbstractFill, dims::Tuple{Vararg{Union{Int,Colon}}}) = 
    fill_reshape(parent, Base._reshape_uncolon(parent, dims)...)
reshape(parent::AbstractFill, shp::Tuple{Union{Integer,Base.OneTo}, Vararg{Union{Integer,Base.OneTo}}}) = 
    reshape(parent, Base.to_shape(shp))    
reshape(parent::AbstractFill, dims::Dims)        = Base._reshape(parent, dims)    
reshape(parent::AbstractFill, dims::Tuple{Integer, Vararg{Integer}})        = Base._reshape(parent, dims)    
Base._reshape(parent::AbstractFill, dims::Dims) = fill_reshape(parent, dims...)
Base._reshape(parent::AbstractFill, dims::Tuple{Integer,Vararg{Integer}}) = fill_reshape(parent, dims...)
# Resolves ambiguity error with `_reshape(v::AbstractArray{T, 1}, dims::Tuple{Int})`
Base._reshape(parent::AbstractFill{T, 1, Axes}, dims::Tuple{Int}) where {T, Axes} = fill_reshape(parent, dims...)

for (Typ, funcs, func) in ((:Zeros, :zeros, :zero), (:Ones, :ones, :one))
    @eval begin
        """ `$($Typ){T, N, Axes} <: AbstractFill{T, N, Axes}` (lazy `$($funcs)` with axes)"""
        struct $Typ{T, N, Axes} <: AbstractFill{T, N, Axes}
            axes::Axes
            @inline $Typ{T,N,Axes}(sz::Axes) where Axes<:Tuple{Vararg{AbstractUnitRange,N}} where {T, N} =
                new{T,N,Axes}(sz)
            @inline $Typ{T,N}(sz::Axes) where Axes<:Tuple{Vararg{AbstractUnitRange,N}} where {T, N} =
                new{T,N,Axes}(sz)
            @inline $Typ{T,0,Tuple{}}(sz::Tuple{}) where T = new{T,0,Tuple{}}(sz)
        end


        @inline $Typ{T, 0}(sz::Tuple{}) where {T} = $Typ{T,0,Tuple{}}(sz)
        @inline $Typ{T, N}(sz::Tuple{Vararg{<:Integer, N}}) where {T, N} = $Typ{T,N}(Base.OneTo.(sz))
        @inline $Typ{T, N}(sz::Vararg{<:Integer, N}) where {T, N} = $Typ{T,N}(sz)
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

        @inline axes(Z::$Typ) = Z.axes
        @inline size(Z::$Typ) = length.(Z.axes)
        @inline getindex_value(Z::$Typ{T}) where T = $func(T)

        AbstractArray{T}(F::$Typ{T}) where T = F
        AbstractArray{T,N}(F::$Typ{T,N}) where {T,N} = F
        AbstractArray{T}(F::$Typ) where T = $Typ{T}(F.axes)
        AbstractArray{T,N}(F::$Typ{V,N}) where {T,V,N} = $Typ{T}(F.axes)
        convert(::Type{AbstractArray{T}}, F::$Typ{T}) where T = AbstractArray{T}(F)
        convert(::Type{AbstractArray{T,N}}, F::$Typ{T,N}) where {T,N} = AbstractArray{T,N}(F)
        convert(::Type{AbstractArray{T}}, F::$Typ) where T = AbstractArray{T}(F)
        convert(::Type{AbstractArray{T,N}}, F::$Typ) where {T,N} = AbstractArray{T,N}(F)

        copy(F::$Typ) = F

        getindex(F::$Typ{T,0}) where T = getindex_value(F)
        function getindex(F::$Typ{T}, kj::Vararg{AbstractVector{II},N}) where {T,II<:Integer,N}
            checkbounds(F, kj...)
            $Typ{T}(length.(kj))
        end
        function getindex(A::$Typ{T}, kr::AbstractVector{Bool}) where T
            length(A) == length(kr) || throw(DimensionMismatch())
            $Typ{T}(count(kr))
        end
        function getindex(A::$Typ{T}, kr::AbstractArray{Bool}) where T
            size(A) == size(kr) || throw(DimensionMismatch())
            $Typ{T}(count(kr))
        end

        function fill_reshape(parent::$Typ{T}, dims::Integer...) where T
            n = length(parent)
            prod(dims) == n || _throw_dmrs(n, "size", dims)
            $Typ{T}(dims...)
        end
    end
end



rank(F::Zeros) = 0
rank(F::Ones) = 1


struct RectDiagonal{T,V<:AbstractVector{T},Axes<:Tuple{Vararg{AbstractUnitRange,2}}} <: AbstractMatrix{T}
    diag::V
    axes::Axes

    @inline function RectDiagonal{T,V}(A::V, axes::Axes) where {T,V<:AbstractVector{T},Axes<:Tuple{Vararg{AbstractUnitRange,2}}}
        @assert !Base.has_offset_axes(A)
        @assert any(length(ax) == length(A) for ax in axes)
        rd = new{T,V,Axes}(A, axes)
        @assert !Base.has_offset_axes(rd)
        return rd
    end
end

@inline RectDiagonal{T,V}(A::V, sz::Tuple{Vararg{Integer, 2}}) where {T,V} = RectDiagonal{T,V}(A, Base.OneTo.(sz))
@inline RectDiagonal{T,V}(A::V, axes::Vararg{Any, 2}) where {T,V} = RectDiagonal{T,V}(A, axes)
@inline RectDiagonal{T,V}(A::V, sz::Vararg{Integer, 2}) where {T,V} = RectDiagonal{T,V}(A, sz)
@inline RectDiagonal{T,V}(A::V) where {T,V} = RectDiagonal{T,V}(A, (axes(A, 1), axes(A, 1)))
@inline RectDiagonal{T}(A::V, args...) where {T,V} = RectDiagonal{T,V}(A, args...)
@inline RectDiagonal(A::V, args...) where {V} = RectDiagonal{eltype(V),V}(A, args...)


# patch missing overload from Base
axes(rd::Diagonal{<:Any,<:AbstractFill}) = (axes(rd.diag,1),axes(rd.diag,1))
axes(T::AbstractTriangular{<:Any,<:AbstractFill}) = axes(parent(T))

axes(rd::RectDiagonal) = rd.axes
size(rd::RectDiagonal) = length.(rd.axes)

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


Base.replace_in_print_matrix(A::RectDiagonal, i::Integer, j::Integer, s::AbstractString) = 
    i == j ? s : Base.replace_with_centered_mark(s)


const RectOrDiagonal{T,V,Axes} = Union{RectDiagonal{T,V,Axes}, Diagonal{T,V}}
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

for f in (:permutedims, :inv, :triu, :triu!, :tril, :tril!)
    @eval ($f)(IM::SquareEye) = IM
end

Eye(n::Integer, m::Integer) = RectDiagonal(Ones(min(n,m)), n, m)
Eye{T}(n::Integer, m::Integer) where T = RectDiagonal{T}(Ones{T}(min(n,m)), n, m)
function Eye{T}((a,b)::NTuple{2,AbstractUnitRange{Int}}) where T 
    ab = length(a) ≤ length(b) ? a : b
    RectDiagonal{T}(Ones{T}((ab,)), (a,b))
end
function Eye((a,b)::NTuple{2,AbstractUnitRange{Int}})
    ab = length(a) ≤ length(b) ? a : b
    RectDiagonal(Ones((ab,)), (a,b))
end


@deprecate Eye{T}(sz::Tuple{Vararg{Integer,2}}) where T Eye{T}(sz...)
@deprecate Eye(sz::Tuple{Vararg{Integer,2}}) Eye{Float64}(sz...)



@inline Eye{T}(A::AbstractMatrix) where T = Eye{T}(size(A)...)
@inline Eye(A::AbstractMatrix) = Eye{eltype(A)}(size(A)...)


#########
#  Special matrix types
#########



## Array
convert(::Type{Array}, F::AbstractFill) = fill(getindex_value(F), size(F))
convert(::Type{Array{T}}, F::AbstractFill) where T = fill(convert(T, getindex_value(F)), size(F))
convert(::Type{Array{T,N}}, F::AbstractFill{V,N}) where {T,V,N} = fill(convert(T, getindex_value(F)), size(F))


# These are in case `zeros` or `ones` are ever faster than `fill`
for (Typ, funcs, func) in ((:Zeros, :zeros, :zero), (:Ones, :ones, :one))
    @eval begin
        convert(::Type{Array}, F::$Typ{T}) where T = $funcs(T, size(F))
        convert(::Type{Array{T}}, F::$Typ{T}) where T = $funcs(T, size(F))
        convert(::Type{Array{T,N}}, F::$Typ{V,N}) where {T,V,N} = $funcs(T,size(F))
    end
end

function convert(::Type{Diagonal}, Z::Zeros{T,2}) where T
    n,m = size(Z)
    n ≠ m && throw(BoundsError(Z))
    Diagonal(zeros(T, n))
end

function convert(::Type{Diagonal{T}}, Z::Zeros{V,2}) where {T,V}
    n,m = size(Z)
    n ≠ m && throw(BoundsError(Z))
    Diagonal(zeros(T, n))
end

## Sparse arrays

convert(::Type{SparseVector}, Z::Zeros{T,1}) where T = spzeros(T, length(Z))
convert(::Type{SparseVector{Tv}}, Z::Zeros{T,1}) where {T,Tv} = spzeros(Tv, length(Z))
convert(::Type{SparseVector{Tv,Ti}}, Z::Zeros{T,1}) where {T,Tv,Ti} = spzeros(Tv, Ti, length(Z))

convert(::Type{AbstractSparseVector}, Z::Zeros{T,1}) where T = spzeros(T, length(Z))
convert(::Type{AbstractSparseVector{Tv}}, Z::Zeros{T,1}) where {Tv,T}= spzeros(Tv, length(Z))

convert(::Type{SparseMatrixCSC}, Z::Zeros{T,2}) where T = spzeros(T, size(Z)...)
convert(::Type{SparseMatrixCSC{Tv}}, Z::Zeros{T,2}) where {T,Tv} = spzeros(Tv, size(Z)...)
convert(::Type{SparseMatrixCSC{Tv,Ti}}, Z::Zeros{T,2}) where {T,Tv,Ti} = spzeros(Tv, Ti, size(Z)...)
convert(::Type{SparseMatrixCSC{Tv,Ti}}, Z::Zeros{T,2,Axes}) where {Tv,Ti<:Integer,T,Axes} =
    spzeros(Tv, Ti, size(Z)...)

convert(::Type{AbstractSparseMatrix}, Z::Zeros{T,2}) where {T} = spzeros(T, size(Z)...)
convert(::Type{AbstractSparseMatrix{Tv}}, Z::Zeros{T,2}) where {T,Tv} = spzeros(Tv, size(Z)...)

convert(::Type{AbstractSparseArray}, Z::Zeros{T}) where T = spzeros(T, size(Z)...)
convert(::Type{AbstractSparseArray{Tv}}, Z::Zeros{T}) where {T,Tv} = spzeros(Tv, size(Z)...)
convert(::Type{AbstractSparseArray{Tv,Ti}}, Z::Zeros{T}) where {T,Tv,Ti} = spzeros(Tv, Ti, size(Z)...)
convert(::Type{AbstractSparseArray{Tv,Ti,N}}, Z::Zeros{T,N}) where {T,Tv,Ti,N} = spzeros(Tv, Ti, size(Z)...)


convert(::Type{SparseMatrixCSC}, Z::Eye{T}) where T = SparseMatrixCSC{T}(I, size(Z)...)
convert(::Type{SparseMatrixCSC{Tv}}, Z::Eye{T}) where {T,Tv} = SparseMatrixCSC{Tv}(I, size(Z)...)
# works around missing `speye`:
convert(::Type{SparseMatrixCSC{Tv,Ti}}, Z::Eye{T}) where {T,Tv,Ti<:Integer} =
    convert(SparseMatrixCSC{Tv,Ti}, SparseMatrixCSC{Tv}(I, size(Z)...))

convert(::Type{AbstractSparseMatrix}, Z::Eye{T}) where {T} = SparseMatrixCSC{T}(I, size(Z)...)
convert(::Type{AbstractSparseMatrix{Tv}}, Z::Eye{T}) where {T,Tv} = SparseMatrixCSC{Tv}(I, size(Z)...)

convert(::Type{AbstractSparseArray}, Z::Eye{T}) where T = SparseMatrixCSC{T}(I, size(Z)...)
convert(::Type{AbstractSparseArray{Tv}}, Z::Eye{T}) where {T,Tv} = SparseMatrixCSC{Tv}(I, size(Z)...)


convert(::Type{AbstractSparseArray{Tv,Ti}}, Z::Eye{T}) where {T,Tv,Ti} =
    convert(SparseMatrixCSC{Tv,Ti}, Z)
convert(::Type{AbstractSparseArray{Tv,Ti,2}}, Z::Eye{T}) where {T,Tv,Ti} =
    convert(SparseMatrixCSC{Tv,Ti}, Z)


#########
# maximum/minimum
#########

for op in (:maximum, :minimum)
    @eval $op(x::AbstractFill) = getindex_value(x)
end


#########
# Cumsum
#########

sum(x::AbstractFill) = getindex_value(x)*length(x)
sum(x::Zeros) = getindex_value(x)

cumsum(x::AbstractFill{<:Any,1}) = range(getindex_value(x); step=getindex_value(x),
                                                    length=length(x))

cumsum(x::Zeros{<:Any,1}) = x
cumsum(x::Zeros{Bool,1}) = x
cumsum(x::Ones{II,1}) where II<:Integer = Base.OneTo{II}(length(x))
cumsum(x::Ones{Bool,1}) = Base.OneTo{Int}(length(x))
cumsum(x::AbstractFill{Bool,1}) = cumsum(convert(AbstractFill{Int}, x))


#########
# Diff
#########

diff(x::AbstractFill{T,1}) where T = Zeros{T}(length(x)-1)

#########
# unique
#########

unique(x::AbstractFill{T}) where T = isempty(x) ? T[] : T[getindex_value(x)]
allunique(x::AbstractFill) = length(x) < 2

#########
# zero
#########

zero(r::Zeros{T,N}) where {T,N} = r
zero(r::Ones{T,N}) where {T,N} = Zeros{T,N}(r.axes)
zero(r::Fill{T,N}) where {T,N} = Zeros{T,N}(r.axes)

#########
# any/all/isone/iszero
#########

function isone(AF::AbstractFill{<:Any,2})
    isone(getindex_value(AF)) || return false
    (n,m) = size(AF)
    n != m && return false
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
# use any/all on scalar to get Boolean error message
any(f::Function, x::AbstractFill) = any(f(getindex_value(x)))
all(f::Function, x::AbstractFill) = all(f(getindex_value(x)))
any(x::AbstractFill) = any(getindex_value(x))
all(x::AbstractFill) = all(getindex_value(x))

count(x::Ones{Bool}) = length(x)
count(x::Zeros{Bool}) = 0
count(f, x::AbstractFill) = f(getindex_value(x)) ? length(x) : 0

include("fillalgebra.jl")
include("fillbroadcast.jl")

##
# print
##
Base.replace_in_print_matrix(::Zeros, ::Integer, ::Integer, s::AbstractString) =
    Base.replace_with_centered_mark(s)

# following support blocked fill array printing via
# BlockArrays.jl
axes_print_matrix_row(_, io, X, A, i, cols, sep) =
    Base.invoke(Base.print_matrix_row, Tuple{IO,AbstractVecOrMat,Vector,Integer,AbstractVector,AbstractString},
                io, X, A, i, cols, sep)

Base.print_matrix_row(io::IO,
        X::Union{AbstractFill{<:Any,1},
                 AbstractFill{<:Any,2},
                 Diagonal{<:Any,<:AbstractFill{<:Any,1}},
                 RectDiagonal,
                 AbstractTriangular{<:Any,<:AbstractFill{<:Any,2}}
                 }, A::Vector,
        i::Integer, cols::AbstractVector, sep::AbstractString) =
        axes_print_matrix_row(axes(X), io, X, A, i, cols, sep)


# Display concise description of a Fill.
Base.show(io::IO, x::AbstractFill) =
    print(io, "$(summary(x)) = $(getindex_value(x))")

function Base.show(io::IO, ::MIME"text/plain", x::AbstractFill)
    show(io, x)
end

end # module
