module FillArraysStatisticsExt

import Statistics: mean, var, cov, cor
using LinearAlgebra: diagind

using FillArrays
using FillArrays: AbstractFill, AbstractFillVector, AbstractFillMatrix, getindex_value

mean(A::AbstractFill; dims=(:)) = mean(identity, A; dims=dims)
function mean(f::Union{Function, Type}, A::AbstractFill; dims=(:))
    val = float(f(getindex_value(A)))
    dims isa Colon ? val :
        Fill(val, ntuple(d -> d in dims ? 1 : size(A,d), ndims(A))...)
end


function var(A::AbstractFill{T}; corrected::Bool=true, mean=nothing, dims=(:)) where {T<:Number}
    dims isa Colon ? zero(float(T)) :
        Zeros{float(T)}(ntuple(d -> d in dims ? 1 : size(A,d), ndims(A))...)
end

cov(::AbstractFillVector{T}; corrected::Bool=true) where {T<:Number} = zero(float(T))
cov(A::AbstractFillMatrix{T}; corrected::Bool=true, dims::Integer=1) where {T<:Number} =
    Zeros{float(T)}(size(A, 3-dims), size(A, 3-dims))

cor(::AbstractFillVector{T}) where {T<:Number} = one(float(T))
function cor(A::AbstractFillMatrix{T}; dims::Integer=1) where {T<:Number}
    out = fill(float(T)(NaN), size(A, 3-dims), size(A, 3-dims))
    out[diagind(out)] .= 1
    out
end

end # module
