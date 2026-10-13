module FillArraysSparseArraysExt

using SparseArrays
using SparseArrays: SparseVectorUnion
import Base: convert, kron
using FillArrays
using FillArrays: RectDiagonal, RectDiagonalFill, OffDiagonal, ZerosVector, ZerosMatrix, AbstractFillVector, _fill_dot,
                  _band, _bandindex, _bandlength
# Specifying the full namespace is necessary because of https://github.com/JuliaLang/julia/issues/48533
# See https://github.com/JuliaStats/LogExpFunctions.jl/pull/63
using FillArrays.LinearAlgebra
import LinearAlgebra: dot, kron, I

##################
## Sparse arrays
##################
SparseVector{T}(Z::ZerosVector) where T = spzeros(T, length(Z))
SparseVector{Tv,Ti}(Z::ZerosVector) where {Tv,Ti} = spzeros(Tv, Ti, length(Z))

convert(::Type{AbstractSparseVector}, Z::ZerosVector{T}) where T = spzeros(T, length(Z))
convert(::Type{AbstractSparseVector{T}}, Z::ZerosVector) where T= spzeros(T, length(Z))

SparseMatrixCSC{T}(Z::ZerosMatrix) where T = spzeros(T, size(Z)...)
SparseMatrixCSC{Tv,Ti}(Z::Zeros{T,2,Axes}) where {Tv,Ti<:Integer,T,Axes} = spzeros(Tv, Ti, size(Z)...)

convert(::Type{AbstractSparseMatrix}, Z::ZerosMatrix{T}) where T = spzeros(T, size(Z)...)
convert(::Type{AbstractSparseMatrix{T}}, Z::ZerosMatrix) where T = spzeros(T, size(Z)...)

convert(::Type{AbstractSparseArray}, Z::Zeros{T}) where T = spzeros(T, size(Z)...)
convert(::Type{AbstractSparseArray{Tv}}, Z::Zeros{T}) where {T,Tv} = spzeros(Tv, size(Z)...)
convert(::Type{AbstractSparseArray{Tv,Ti}}, Z::Zeros{T}) where {T,Tv,Ti} = spzeros(Tv, Ti, size(Z)...)
convert(::Type{AbstractSparseArray{Tv,Ti,N}}, Z::Zeros{T,N}) where {T,Tv,Ti,N} = spzeros(Tv, Ti, size(Z)...)

SparseMatrixCSC{Tv}(Z::Eye{T}) where {T,Tv} = SparseMatrixCSC{Tv}(I, size(Z)...)
# works around missing `speye`:
SparseMatrixCSC{Tv,Ti}(Z::Eye{T}) where {T,Tv,Ti<:Integer} =
    convert(SparseMatrixCSC{Tv,Ti}, SparseMatrixCSC{Tv}(I, size(Z)...))

convert(::Type{AbstractSparseMatrix}, Z::Eye{T}) where {T} = SparseMatrixCSC{T}(I, size(Z)...)
convert(::Type{AbstractSparseMatrix{Tv}}, Z::Eye{T}) where {T,Tv} = SparseMatrixCSC{Tv}(I, size(Z)...)

convert(::Type{AbstractSparseArray}, Z::Eye{T}) where T = SparseMatrixCSC{T}(I, size(Z)...)
convert(::Type{AbstractSparseArray{Tv}}, Z::Eye{T}) where {T,Tv} = SparseMatrixCSC{Tv}(I, size(Z)...)


convert(::Type{AbstractSparseArray{Tv,Ti}}, Z::Eye{T}) where {T,Tv,Ti} =
    convert(SparseMatrixCSC{Tv,Ti}, Z)
convert(::Type{AbstractSparseArray{Tv,Ti,2}}, Z::Eye{T}) where {T,Tv,Ti} =
    convert(SparseMatrixCSC{Tv,Ti}, Z)

const SparseSingleBandMatrix = Union{Diagonal{<:Any,<:AbstractFillVector}, RectDiagonal, OffDiagonal}

function SparseMatrixCSC{Tv}(A::SparseSingleBandMatrix) where {Tv}
    SparseMatrixCSC{Tv,eltype(axes(A,1))}(A)
end
# only the band is traversed, and zeros in the band are not stored, as for Diagonal
function SparseMatrixCSC{Tv,Ti}(A::SparseSingleBandMatrix) where {Tv,Ti}
    Base.require_one_based_indexing(A)
    m, n = size(A)
    k = _bandindex(A)
    band = _band(A)
    len = _bandlength(m, n, k)
    # entry t of the band lies in row r₀ + t and column c₀ + t
    r₀, c₀ = max(0, -k), max(0, k)
    colptr = Vector{Ti}(undef, n + 1)
    rowval = sizehint!(Ti[], len)
    nzval = sizehint!(Tv[], len)
    colptr[1] = 1
    for j in 1:n
        t = j - c₀
        if 1 ≤ t ≤ len
            v = band[t]
            if SparseArrays._isnotzero(v)
                push!(rowval, r₀ + t)
                push!(nzval, v)
            end
        end
        colptr[j+1] = length(nzval) + 1
    end
    SparseMatrixCSC{Tv,Ti}(m, n, colptr, rowval, nzval)
end

# TODO: remove in v2.0
@deprecate kron(E1::RectDiagonalFill, E2::RectDiagonalFill) kron(sparse(E1), sparse(E2))

# Ambiguity. see #178
dot(x::AbstractFillVector, y::SparseVectorUnion) = _fill_dot(x, y)

end # module
