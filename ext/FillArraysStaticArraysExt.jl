module FillArraysStaticArraysExt

using FillArrays
using StaticArrays

import Base: promote_op
import FillArrays: elconvert, has_mutable_storage

# An `SArray` holds its entries in the type's own immutable storage, so a broadcast that leaves
# them alone may return it. Not every `StaticArray` may: an `MArray` is assignable, and a
# `SizedArray` wraps an ordinary one.
has_mutable_storage(::SArray) = false

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