module FillArraysStaticArraysExt

using FillArrays
using StaticArrays

import Base: promote_op
import FillArrays: elconvert

# Disambiguity methods for StaticArrays

function Base.:+(a::FillArrays.Zeros, b::StaticArray)
    promote_shape(a,b)
    return elconvert(promote_op(+,eltype(a),eltype(b)),b)
end
function Base.:+(a::StaticArray, b::FillArrays.Zeros)
    promote_shape(a,b)
    return elconvert(promote_op(+,eltype(a),eltype(b)),a)
end
function Base.:-(a::StaticArray, b::FillArrays.Zeros)
    promote_shape(a,b)
    return elconvert(promote_op(-,eltype(a),eltype(b)),a)
end
function Base.:-(a::FillArrays.Zeros, b::StaticArray)
    promote_shape(a,b)
    return elconvert(promote_op(-,eltype(a),eltype(b)),-b)
end

# slicing a SingleBandMatrix with static axes uses the generic fallback, which returns a static array
Base.@propagate_inbounds FillArrays._singleband_getindex(::Tuple{SOneTo,Any}, A, kr, jr) = invoke(getindex, Tuple{AbstractArray, Vararg{Any}}, A, kr, jr)
Base.@propagate_inbounds FillArrays._singleband_getindex(::Tuple{Any,SOneTo}, A, kr, jr) = invoke(getindex, Tuple{AbstractArray, Vararg{Any}}, A, kr, jr)
Base.@propagate_inbounds FillArrays._singleband_getindex(::Tuple{SOneTo,SOneTo}, A, kr, jr) = invoke(getindex, Tuple{AbstractArray, Vararg{Any}}, A, kr, jr)

end # module