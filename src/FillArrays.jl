__precompile__()
module FillArrays
using Compat
using Compat.LinearAlgebra, Compat.SparseArrays
import Base: size, getindex, setindex!, IndexStyle, checkbounds, convert,
                +, -, *, /, \, sum, cumsum, maximum, minimum
import Compat.LinearAlgebra: rank
import Compat: AbstractRange

if VERSION ≥ v"0.7"
    import Base.Broadcast: broadcasted, DefaultArrayStyle
end


export Zeros, Ones, Fill, Eye

abstract type AbstractFill{T, N} <: AbstractArray{T, N} end


@inline function getindex(F::AbstractFill, k::Integer)
    @boundscheck checkbounds(F, k)
    getindex_value(F)
end

@inline function getindex(F::AbstractFill{T, N}, kj::Vararg{<:Integer, N}) where {T, N}
    @boundscheck checkbounds(F, kj...)
    getindex_value(F)
end

rank(F::AbstractFill) = iszero(getindex_value(F)) ? 0 : 1

IndexStyle(::Type{<:AbstractFill}) = IndexLinear()


struct Fill{T, N, SZ} <: AbstractFill{T, N}
    value::T
    size::SZ

    @inline function Fill{T,N,SZ}(x::T, sz::SZ) where SZ<:Tuple{Vararg{Integer,N}} where {T, N}
        @boundscheck any(k -> k < 0, sz) && throw(BoundsError())
        new{T,N,SZ}(x,sz)
    end
    @inline Fill{T, N}(x::T, sz::SZ) where SZ<:Tuple{Vararg{Integer,N}} where {T, N} = Fill{T,N,SZ}(x, sz)
    @inline Fill{T, N}(x, sz::SZ) where SZ<:Tuple{Vararg{Integer,N}} where {T, N} = Fill{T,N}(convert(T, x)::T, sz)
    @inline Fill{T, N}(x, sz::Vararg{Integer, N}) where {T, N} = Fill{T,N}(convert(T, x)::T, sz)
end


@inline Fill{T}(x, sz::Vararg{<:Integer,N}) where {T, N} = Fill{T, N}(x, sz)
@inline Fill{T}(x, sz::Tuple{Vararg{Integer,N}}) where {T, N} = Fill{T, N}(x, sz)
@inline Fill(x::T, sz::Vararg{<:Integer,N}) where {T, N}  = Fill{T, N}(x, sz)
@inline Fill(x::T, sz::Tuple{Vararg{Integer,N}}) where {T, N}  = Fill{T, N}(x, sz)

@inline size(F::Fill) = F.size
@inline getindex_value(F::Fill) = F.value

AbstractArray{T}(F::Fill{T}) where T = F
AbstractArray{T,N}(F::Fill{T,N}) where {T,N} = F
AbstractArray{T}(F::Fill{V,N}) where {T,V,N} = Fill{T}(convert(T, F.value)::T, F.size)
AbstractArray{T,N}(F::Fill{V,N}) where {T,V,N} = Fill{T}(convert(T, F.value)::T, F.size)

convert(::Type{AbstractArray{T}}, F::Fill{T}) where T = F
convert(::Type{AbstractArray{T,N}}, F::Fill{T,N}) where {T,N} = F
convert(::Type{AbstractArray{T}}, F::Fill) where {T} = AbstractArray{T}(F)
convert(::Type{AbstractArray{T,N}}, F::Fill) where {T,N} = AbstractArray{T,N}(F)
convert(::Type{AbstractFill}, F::AbstractFill) = F
convert(::Type{AbstractFill{T}}, F::AbstractFill) where T = convert(AbstractArray{T}, F)
convert(::Type{AbstractFill{T,N}}, F::AbstractFill) where {T,N} = convert(AbstractArray{T,N}, F)


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

+(a::AbstractFill) = a
-(a::AbstractFill) = Fill(-getindex_value(a), size(a))

# Fill +/- Fill
function +(a::AbstractFill{T, N}, b::AbstractFill{V, N}) where {T, V, N}
    size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
    return Fill(getindex_value(a) + getindex_value(b), size(a))
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
        struct $Typ{T, N, SZ} <: AbstractFill{T, N}
            size::SZ
            @inline function $Typ{T, N}(sz::SZ) where SZ<:Tuple{Vararg{Integer,N}} where {T, N}
                @boundscheck any(k -> k < 0, sz) && throw(BoundsError())
                new{T,N,SZ}(sz)
            end
            @inline $Typ{T, N}(sz::Vararg{<:Integer, N}) where {T, N} = $Typ(sz)
        end

        @inline $Typ{T}(sz::Vararg{Integer,N}) where {T, N} = $Typ{T, N}(sz)
        @inline $Typ{T}(sz::SZ) where SZ<:Tuple{Vararg{Integer,N}} where {T, N} = $Typ{T, N}(sz)
        @inline $Typ(sz::Vararg{Integer,N}) where N = $Typ{Float64,N}(sz)
        @inline $Typ(sz::SZ) where SZ<:Tuple{Vararg{Integer,N}} where N = $Typ{Float64,N}(sz)

        @inline $Typ{T,N}(A::AbstractArray{V,N}) where{T,V,N} = $Typ{T,N}(size(A))
        @inline $Typ{T}(A::AbstractArray) where{T} = $Typ{T}(size(A))
        @inline $Typ(A::AbstractArray) = $Typ(size(A))

        @inline size(Z::$Typ) = Z.size
        @inline getindex_value(Z::$Typ{T}) where T = $func(T)

        AbstractArray{T}(F::$Typ{T}) where T = F
        AbstractArray{T,N}(F::$Typ{T,N}) where {T,N} = F
        AbstractArray{T}(F::$Typ) where T = $Typ{T}(F.size)
        AbstractArray{T,N}(F::$Typ{V,N}) where {T,V,N} = $Typ{T}(F.size)
        convert(::Type{AbstractArray{T}}, F::$Typ{T}) where T = AbstractArray{T}(F)
        convert(::Type{AbstractArray{T,N}}, F::$Typ{T,N}) where {T,N} = AbstractArray{T,N}(F)
        convert(::Type{AbstractArray{T}}, F::$Typ) where T = AbstractArray{T}(F)
        convert(::Type{AbstractArray{T,N}}, F::$Typ) where {T,N} = AbstractArray{T,N}(F)

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




struct Eye{T, SZ} <: AbstractMatrix{T}
    size::SZ
    @inline function Eye{T}(sz::SZ) where {T,SZ<:Tuple{Vararg{Integer,2}}}
        @boundscheck any(k -> k < 0, sz) && throw(BoundsError())
        new{T,SZ}(sz)
    end

    Eye{T}(sz::Vararg{Integer,2}) where {T} = Eye{T}(sz)
end

Eye{T}(n::Integer) where T = Eye{T}(n, n)
Eye(n::Integer, m::Integer) = Eye{Float64}(n, m)
Eye(sz::Tuple{Vararg{Integer,2}}) = Eye{Float64}(sz)
Eye(n::Integer) = Eye(n, n)

@inline Eye{T}(A::AbstractMatrix) where T = Eye{T}(size(A))
@inline Eye(A::AbstractMatrix) = Eye{eltype(A)}(size(A))

size(E::Eye) = E.size
rank(E::Eye) = minimum(size(E))

@inline function getindex(E::Eye{T}, k::Integer, j::Integer) where T
    @boundscheck checkbounds(E, k, j)
    ifelse(k == j, one(T), zero(T))
end

IndexStyle(::Type{<:Eye}) = IndexCartesian()

AbstractArray{T}(E::Eye{T}) where T = E
AbstractMatrix{T}(E::Eye{T}) where T = E
AbstractArray{T}(E::Eye) where T = Eye{T}(E.size)
AbstractMatrix{T}(E::Eye) where T = Eye{T}(E.size)
convert(::Type{AbstractArray{T}}, E::Eye{T}) where T = E
convert(::Type{AbstractMatrix{T}}, E::Eye{T}) where T = E
convert(::Type{AbstractArray{T}}, E::Eye) where T = Eye{T}(E.size)
convert(::Type{AbstractMatrix{T}}, E::Eye) where T = Eye{T}(E.size)



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

if VERSION < v"0.7.0-DEV.2565"
    convert(::Type{Array},     E::Eye{T}) where T = eye(T, E.size[1], E.size[2])
    convert(::Type{Array{T}},  E::Eye)    where T = eye(T, E.size[1], E.size[2])
    convert(::Type{Matrix{T}}, E::Eye)    where T = eye(T, E.size[1], E.size[2])
else
    convert(::Type{Array},     E::Eye{T}) where T = Matrix{T}(I, E.size[1], E.size[2])
    convert(::Type{Array{T}},  E::Eye)    where T = Matrix{T}(I, E.size[1], E.size[2])
    convert(::Type{Matrix{T}}, E::Eye)    where T = Matrix{T}(I, E.size[1], E.size[2])
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


function convert(::Type{Diagonal}, E::Eye{T}) where T
    n,m = size(E)
    n ≠ m && throw(BoundsError(E))
    Diagonal(ones(T, n))
end

function convert(::Type{Diagonal{T}}, E::Eye{V}) where {T,V}
    n,m = size(E)
    n ≠ m && throw(BoundsError(E))
    Diagonal(ones(T, n))
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


if VERSION < v"0.7.0-DEV.2565"
    convert(::Type{SparseMatrixCSC}, Z::Eye{T}) where T = speye(T, size(Z)...)
    convert(::Type{SparseMatrixCSC{Tv}}, Z::Eye{T}) where {T,Tv} = speye(Tv, size(Z)...)
    # works around missing `speye`:
    convert(::Type{SparseMatrixCSC{Tv,Ti}}, Z::Eye{T}) where {T,Tv,Ti} =
        convert(SparseMatrixCSC{Tv,Ti}, speye(Tv, size(Z)...))

    convert(::Type{AbstractSparseMatrix}, Z::Eye{T}) where {T} = speye(T, size(Z)...)
    convert(::Type{AbstractSparseMatrix{Tv}}, Z::Eye{T}) where {T,Tv} = speye(Tv, size(Z)...)

    convert(::Type{AbstractSparseArray}, Z::Eye{T}) where T = speye(T, size(Z)...)
    convert(::Type{AbstractSparseArray{Tv}}, Z::Eye{T}) where {T,Tv} = speye(Tv, size(Z)...)
else
    convert(::Type{SparseMatrixCSC}, Z::Eye{T}) where T = SparseMatrixCSC{T}(I, size(Z)...)
    convert(::Type{SparseMatrixCSC{Tv}}, Z::Eye{T}) where {T,Tv} = SparseMatrixCSC{Tv}(I, size(Z)...)
    # works around missing `speye`:
    convert(::Type{SparseMatrixCSC{Tv,Ti}}, Z::Eye{T}) where {T,Tv,Ti<:Integer} =
        convert(SparseMatrixCSC{Tv,Ti}, SparseMatrixCSC{Tv}(I, size(Z)...))

    convert(::Type{AbstractSparseMatrix}, Z::Eye{T}) where {T} = SparseMatrixCSC{T}(I, size(Z)...)
    convert(::Type{AbstractSparseMatrix{Tv}}, Z::Eye{T}) where {T,Tv} = SparseMatrixCSC{Tv}(I, size(Z)...)

    convert(::Type{AbstractSparseArray}, Z::Eye{T}) where T = SparseMatrixCSC{T}(I, size(Z)...)
    convert(::Type{AbstractSparseArray{Tv}}, Z::Eye{T}) where {T,Tv} = SparseMatrixCSC{Tv}(I, size(Z)...)
end

convert(::Type{AbstractSparseArray{Tv,Ti}}, Z::Eye{T}) where {T,Tv,Ti} =
    convert(SparseMatrixCSC{Tv,Ti}, Z)
convert(::Type{AbstractSparseArray{Tv,Ti,2}}, Z::Eye{T}) where {T,Tv,Ti} =
    convert(SparseMatrixCSC{Tv,Ti}, Z)

## Algebraic identities

function mult_zeros(a, b::AbstractMatrix)
    size(a, 2) ≠ size(b, 1) &&
        throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
    return Zeros{promote_type(eltype(a), eltype(b))}(size(a, 1), size(b, 2))
end
function mult_zeros(a, b::AbstractVector)
    size(a, 2) ≠ size(b, 1) &&
        throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
    return Zeros{promote_type(eltype(a), eltype(b))}(size(a, 1))
end

const ZerosVecOrMat{T} = Union{Zeros{T,1}, Zeros{T,2}}
*(a::ZerosVecOrMat, b::AbstractMatrix) = mult_zeros(a, b)
*(a::AbstractMatrix, b::ZerosVecOrMat) = mult_zeros(a, b)
*(a::ZerosVecOrMat, b::AbstractVector) = mult_zeros(a, b)
*(a::AbstractVector, b::ZerosVecOrMat) = mult_zeros(a, b)
*(a::ZerosVecOrMat, b::ZerosVecOrMat) = mult_zeros(a, b)

if VERSION >= v"0.7"
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
end

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
if VERSION < v"0.7-"
    copy_convert(::Type{T}, ::Type{T}, b) where T = copy(b)
    copy_convert(::Type{T}, ::Type{V}, b) where {T,V} = AbstractArray{V}(b)

    function +(a::Zeros{T, N}, b::Array{V, N}) where {T, V, N}
        size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
        return copy_convert(V, promote_type(T,V), b)
    end
    function +(a::Array{T, N}, b::Zeros{V, N}) where {T, V, N}
        size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
        return copy_convert(T, promote_type(T,V), a)
    end

else
    function +(a::Zeros{T, N}, b::Array{V, N}) where {T, V, N}
        size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
        return AbstractArray{promote_type(T,V),N}(b)
    end
    function +(a::Array{T, N}, b::Zeros{V, N}) where {T, V, N}
        size(a) ≠ size(b) && throw(DimensionMismatch("dimensions must match."))
        return AbstractArray{promote_type(T,V),N}(a)
    end
end

function -(a::Zeros{T, N}, b::Array{V, N}) where {T, V, N}
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


#########
# broadcasting
#########

if VERSION ≥ v"0.7"
    for op in (:+, :-)
        @eval broadcasted(::DefaultArrayStyle{N}, ::typeof($op), r1::AbstractFill{T,N}, r2::AbstractFill{V,N}) where {T,V,N} =
            $op(r1, r2)
    end

    broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}) where {T,N} = Fill(op(getindex_value(r)), size(r))
    broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}, x::Number) where {T,N} = Fill(op(getindex_value(r),x), size(r))
    broadcasted(::DefaultArrayStyle{N}, op, x::Number, r::AbstractFill{T,N}) where {T,N} = Fill(op(x, getindex_value(r)), size(r))
    function broadcasted(::DefaultArrayStyle{N}, op, r1::AbstractFill{T,N}, r2::AbstractFill{V,N}) where {T,V,N}
        size(r1) ≠ size(r2) && throw(DimensionMismatch("dimensions must match."))
        Fill(op(getindex_value(r1),getindex_value(r2)), size(r1))
    end

    for op in (:*, :/, :\)
        @eval function broadcasted(::DefaultArrayStyle{N}, ::typeof($op), r1::Ones{T,N}, r2::Ones{V,N}) where {T,V,N}
            size(r1) ≠ size(r2) && throw(DimensionMismatch("dimensions must match."))
            Ones{promote_type(T,V)}(size(r1))
        end
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

sum(x::AbstractFill) = getindex_value(x)*length(x)
sum(x::Zeros) = getindex_value(x)

if VERSION ≥ v"0.7"
    cumsum(x::AbstractFill) = range(getindex_value(x); step=getindex_value(x),
                                                        length=length(x))
else
    cumsum(x::AbstractFill) = range(getindex_value(x), getindex_value(x),
                                                        length(x))
end

cumsum(x::Zeros) = x
cumsum(x::Zeros{Bool}) = x
cumsum(x::Ones{II}) where II<:Integer = Base.OneTo{II}(length(x))
cumsum(x::Ones{Bool}) = Base.OneTo{Int}(length(x))
cumsum(x::AbstractFill{Bool}) = cumsum(convert(AbstractFill{Int}, x))

end # module
