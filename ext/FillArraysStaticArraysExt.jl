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

end # module