__precompile__()
module FillArrays
if VERSION ≥ v"0.7-"
    using LinearAlgebra, SparseArrays
end
import Base: size, getindex, setindex!, IndexStyle, checkbounds, convert

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

IndexStyle(F::AbstractFill) = IndexLinear()


struct Fill{T, N} <: AbstractFill{T, N}
    value::T
    size::NTuple{N, Int}

    @inline function Fill{T, N}(x::T, sz::NTuple{N, Int}) where {T, N}
        @boundscheck any(k -> k < 0, sz) && throw(BoundsError())
        new{T,N}(x,sz)
    end
    @inline Fill{T, N}(x::T, sz::Vararg{Int, N}) where {T, N} = Fill{T,N}(x, sz)
    @inline Fill{T, N}(x, sz::NTuple{N, Int}) where {T, N} = new{T, N}(convert(T, x)::T, sz)
    @inline Fill{T, N}(x, sz::Vararg{Int, N}) where {T, N} = new{T, N}(convert(T, x)::T, sz)
end


@inline Fill{T}(x, sz::Vararg{Int, N}) where {T, N} = Fill{T, N}(x, sz)
@inline Fill{T}(x, sz::NTuple{N, Int}) where {T, N} = Fill{T, N}(x, sz)
@inline Fill(x::T, sz::Vararg{Int,N}) where {T, N}  = Fill{T, N}(x, sz)
@inline Fill(x::T, sz::NTuple{N,Int}) where {T, N}  = Fill{T, N}(x, sz)

@inline size(F::Fill) = F.size
@inline getindex_value(F::Fill) = F.value

convert(::Type{AbstractArray{T}}, F::Fill{T}) where T = F
convert(::Type{AbstractArray{T,N}}, F::Fill{T,N}) where {T,N} = F

convert(::Type{AbstractArray{T}}, F::Fill{V,N}) where {T,V,N} = Fill{T}(convert(T, F.value)::T, F.size)
convert(::Type{AbstractArray{T,N}}, F::Fill{V,N}) where {T,V,N} = Fill{T}(convert(T, F.value)::T, F.size)

for (Typ, funcs, func) in ((:Zeros, :zeros, :zero), (:Ones, :ones, :one))
    @eval begin
        struct $Typ{T, N} <: AbstractFill{T, N}
            size::NTuple{N, Int}
            @inline function $Typ{T, N}(sz::NTuple{N, Int}) where {T, N}
                @boundscheck any(k -> k < 0, sz) && throw(BoundsError())
                new{T,N}(sz)
            end
            @inline $Typ{T, N}(sz::Vararg{Int, N}) where {T, N} = $Typ(sz)
        end

        @inline $Typ{T}(sz::Vararg{Int, N}) where {T, N} = $Typ{T, N}(sz)
        @inline $Typ{T}(sz::NTuple{N, Int}) where {T, N} = $Typ{T, N}(sz)
        @inline $Typ(sz::Vararg{Int,N}) where N = $Typ{Float64,N}(sz)
        @inline $Typ(sz::NTuple{N,Int}) where N = $Typ{Float64,N}(sz)

        @inline $Typ{T,N}(A::AbstractArray{V,N}) where{T,V,N} = $Typ{T,N}(size(A))
        @inline $Typ{T}(A::AbstractArray) where{T} = $Typ{T}(size(A))
        @inline $Typ(A::AbstractArray) = $Typ(size(A))

        @inline size(Z::$Typ) = Z.size
        @inline getindex_value(Z::$Typ{T}) where T = $func(T)

        convert(::Type{AbstractArray{T}}, F::$Typ{T}) where T = F
        convert(::Type{AbstractArray{T,N}}, F::$Typ{T,N}) where {T,N} = F

        convert(::Type{AbstractArray{T}}, F::$Typ) where T = $Typ{T}(F.size)
        convert(::Type{AbstractArray{T,N}}, F::$Typ{V,N}) where {T,V,N} = $Typ{T}(F.size)
    end
end


struct Eye{T} <: AbstractMatrix{T}
    size::NTuple{2, Int}
    @inline function Eye{T}(sz::NTuple{2, Int}) where {T}
        @boundscheck any(k -> k < 0, sz) && throw(BoundsError())
        new{T}(sz)
    end

    Eye{T}(sz::Vararg{Int, 2}) where {T} = Eye{T}(sz)
end

Eye{T}(n::Int) where T = Eye{T}(n, n)
Eye(n::Int, m::Int) = Eye{Float64}(n, m)
Eye(sz::NTuple{2, Int}) = Eye{Float64}(sz)
Eye(n::Int) = Eye(n, n)

@inline Eye{T}(A::AbstractMatrix) where T = Eye{T}(size(A))
@inline Eye(A::AbstractMatrix) = Eye{eltype(A)}(size(A))

size(E::Eye) = E.size

@inline function getindex(E::Eye{T}, k::Integer, j::Integer) where T
    @boundscheck checkbounds(E, k, j)
    ifelse(k == j, one(T), zero(T))
end

IndexStyle(E::Eye) = IndexCartesian()

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
    convert(::Type{SparseMatrixCSC{Tv,Ti}}, Z::Eye{T}) where {T,Tv,Ti} =
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

end # module
