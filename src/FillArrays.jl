module FillArrays

using LinearAlgebra, SparseArrays
import Base: size, getindex, setindex!, IndexStyle, checkbounds, convert,
    +, -, *, /, sum, cumsum, maximum, minimum, sort, sort!,
    any, all, axes, isone, iterate, unique, allunique, permutedims, inv

import Base.\

import LinearAlgebra: rank, svdvals!, tril, triu, tril!, triu!

import Base.Broadcast: broadcasted, DefaultArrayStyle, broadcast_shape



export Zeros, Ones, Fill, Eye

abstract type AbstractFill{T, N, Axes} <: AbstractArray{T, N} end


@inline function getindex(F::AbstractFill, k::Integer)
    @boundscheck checkbounds(F, k)
    getindex_value(F)
end

@inline function getindex(F::AbstractFill{T, N}, kj::Vararg{<:Integer, N}) where {T, N}
    @boundscheck checkbounds(F, kj...)
    getindex_value(F)
end

rank(F::AbstractFill) = iszero(getindex_value(F)) ? 0 : 1
IndexStyle(::Type{<:AbstractFill{<:Any,N,<:NTuple{N,Base.OneTo{Int}}}}) where N = IndexLinear()


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
@inline Fill(x::T, sz::Vararg{<:Integer,N}) where {T, N}  = Fill{T, N}(x, sz)
@inline Fill(x::T, sz::Tuple{Vararg{<:Any,N}}) where {T, N}  = Fill{T, N}(x, sz)

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



for (Typ, funcs, func) in ((:Zeros, :zeros, :zero), (:Ones, :ones, :one))
    @eval begin
        struct $Typ{T, N, Axes} <: AbstractFill{T, N, Axes}
            axes::Axes
            @inline $Typ{T, N}(sz::Axes) where Axes<:Tuple{Vararg{AbstractUnitRange,N}} where {T, N} =
                new{T,N,Axes}(sz)
            @inline $Typ{T,0,Tuple{}}(sz::Tuple{}) where T = new{T,0,Tuple{}}(sz)
        end


        @inline $Typ{T, 0}(sz::Tuple{}) where {T} = $Typ{T,0,Tuple{}}(sz)
        @inline $Typ{T, N}(sz::Tuple{Vararg{<:Integer, N}}) where {T, N} = $Typ{T,N}(Base.OneTo.(sz))
        @inline $Typ{T, N}(sz::Vararg{<:Integer, N}) where {T, N} = $Typ{T,N}(sz)
        @inline $Typ{T}(sz::Vararg{Integer,N}) where {T, N} = $Typ{T, N}(sz)
        @inline $Typ{T}(sz::SZ) where SZ<:Tuple{Vararg{Any,N}} where {T, N} = $Typ{T, N}(sz)
        @inline $Typ(sz::Vararg{Any,N}) where N = $Typ{Float64,N}(sz)
        @inline $Typ(sz::SZ) where SZ<:Tuple{Vararg{Any,N}} where N = $Typ{Float64,N}(sz)

        @inline $Typ{T,N}(A::AbstractArray{V,N}) where{T,V,N} = $Typ{T,N}(size(A))
        @inline $Typ{T}(A::AbstractArray) where{T} = $Typ{T}(size(A))
        @inline $Typ(A::AbstractArray) = $Typ(size(A))

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
    end
end



rank(F::Zeros) = 0
rank(F::Ones) = 1


const Eye{T, Axes} = Diagonal{T, Ones{T,1,Tuple{Axes}}}

Eye{T}(n::Integer) where T = Diagonal(Ones{T}(n))
Eye(n::Integer) = Diagonal(Ones(n))

function iterate(iter::Eye, istate = (1, 1))
    (i::Int, j::Int) = istate
    m = size(iter, 1)
    return i > m ? nothing :
        ((@inbounds getindex(iter, i, j)),
         j == m ? (i + 1, 1) : (i, j + 1))
end

isone(::Eye) = true

for f in (:permutedims, :triu, :triu!, :tril, :tril!, :inv)
    @eval ($f)(IM::Eye) = IM
end

@deprecate Eye(n::Integer, m::Integer) view(Eye(max(n,m)), 1:n, 1:m)
@deprecate Eye{T}(n::Integer, m::Integer) where T view(Eye{T}(max(n,m)), 1:n, 1:m)
@deprecate Eye{T}(sz::Tuple{Vararg{Integer,2}}) where T Eye{T}(sz...)
@deprecate Eye(sz::Tuple{Vararg{Integer,2}}) Eye{Float64}(sz...)

@inline Eye{T}(A::AbstractMatrix) where T = Eye{T}(size(A))
@inline Eye(A::AbstractMatrix) = Eye{eltype(A)}(size(A))


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

cumsum(x::AbstractFill) = range(getindex_value(x); step=getindex_value(x),
                                                    length=length(x))

cumsum(x::Zeros) = x
cumsum(x::Zeros{Bool}) = x
cumsum(x::Ones{II}) where II<:Integer = Base.OneTo{II}(length(x))
cumsum(x::Ones{Bool}) = Base.OneTo{Int}(length(x))
cumsum(x::AbstractFill{Bool}) = cumsum(convert(AbstractFill{Int}, x))

#########
# unique
#########

unique(x::AbstractFill{T}) where T = isempty(x) ? T[] : T[getindex_value(x)]
allunique(x::AbstractFill) = length(x) < 2


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
    d = size(IM, 1)
    d > 1 && return f(zero(T)) || f(one(T))
    d == 1 && return f(one(T))
    return false
end

function all(f::Function, IM::Eye{T}) where T
    d = size(IM, 1)
    d > 1 && return f(zero(T)) && f(one(T))
    d == 1 && return f(one(T))
    return false
end

# In particular, these make iszero(Eye(n))  efficient.
any(f::Function, x::AbstractFill) = f(getindex_value(x))
all(f::Function, x::AbstractFill) = f(getindex_value(x))

include("fillalgebra.jl")
include("fillbroadcast.jl")

end # module
