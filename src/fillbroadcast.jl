### Binary broadcasting implementation.

function broadcasted(::DefaultArrayStyle, op, a::AbstractFill, b::AbstractFill)
    val = op(getindex_value(a), getindex_value(b))
    return Fill(val, broadcast_shape(size(a), size(b)))
end

function _broadcasted_zeros(a, b)
    return Zeros{promote_type(eltype(a), eltype(b))}(broadcast_shape(size(a), size(b)))
end
function _broadcasted_ones(a, b)
    return Ones{promote_type(eltype(a), eltype(b))}(broadcast_shape(size(a), size(b)))
end

broadcasted(::DefaultArrayStyle, ::typeof(+), a::Zeros, b::Zeros) = _broadcasted_zeros(a, b)
broadcasted(::DefaultArrayStyle, ::typeof(+), a::Ones, b::Zeros) = _broadcasted_ones(a, b)
broadcasted(::DefaultArrayStyle, ::typeof(+), a::Zeros, b::Ones) = _broadcasted_ones(a, b)

broadcasted(::DefaultArrayStyle, ::typeof(*), a::Zeros, b::Zeros) = _broadcasted_zeros(a, b)

broadcasted(::DefaultArrayStyle, ::typeof(*), a::Zeros, b::Ones) = _broadcasted_zeros(a, b)
broadcasted(::DefaultArrayStyle, ::typeof(*), a::Zeros, b::Fill) = _broadcasted_zeros(a, b)
function broadcasted(::DefaultArrayStyle, ::typeof(*), a::Zeros, b::AbstractRange)
    return _broadcasted_zeros(a, b)
end
function broadcasted(::DefaultArrayStyle, ::typeof(*), a::Zeros, b::AbstractArray)
    return _broadcasted_zeros(a, b)
end

broadcasted(::DefaultArrayStyle, ::typeof(*), a::Ones, b::Zeros) = _broadcasted_zeros(a, b)
broadcasted(::DefaultArrayStyle, ::typeof(*), a::Fill, b::Zeros) = _broadcasted_zeros(a, b)
function broadcasted(::DefaultArrayStyle, ::typeof(*), a::AbstractRange, b::Zeros)
    return _broadcasted_zeros(a, b)
end
function broadcasted(::DefaultArrayStyle, ::typeof(*), a::AbstractArray, b::Zeros)
    return _broadcasted_zeros(a, b)
end

broadcasted(::DefaultArrayStyle, ::typeof(*), a::Ones, b::Ones) = _broadcasted_ones(a, b)
broadcasted(::DefaultArrayStyle, ::typeof(/), a::Ones, b::Ones) = _broadcasted_ones(a, b)
broadcasted(::DefaultArrayStyle, ::typeof(\), a::Ones, b::Ones) = _broadcasted_ones(a, b)

# function broadcasted(::DefaultArrayStyle{N}, ::typeof(*), a::Zeros{T,N}, b::Zeros{V,N}) where {T,V,N}
#     axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
#     Zeros{promote_type(T,V)}(axes(a))
# end

# function _broadcasted_mul(a::AbstractArray{T}, b::Zeros{V}) where {T,V}
#     axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
#     Zeros{promote_type(T,V)}(axes(a))
# end
# function broadcasted(::DefaultArrayStyle{N}, ::typeof(*), a::AbstractArray{T,N}, b::Zeros{V,N}) where {T,V,N}
#     return _broadcasted_mul(a, b)
# end
# function broadcasted(::DefaultArrayStyle{N}, ::typeof(*), a::AbstractFill{T,N}, b::Zeros{V,N}) where {T,V,N}
#     return _broadcasted_mul(a, b)
# end

# function _broadcasted_mul(a::Zeros{T}, b::AbstractArray{V}) where {T,V}
#     axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
#     Zeros{promote_type(T,V)}(axes(a))
# end
# function broadcasted(::DefaultArrayStyle{N}, ::typeof(*), a::Zeros{T,N}, b::AbstractArray{V,N}) where {T,V,N}
#     _broadcasted_mul(a, b)
# end
# function broadcasted(::DefaultArrayStyle{N}, ::typeof(*), a::Zeros{T,N}, b::AbstractFill{V,N}) where {T,V,N}
#     _broadcasted_mul(a, b)
# end

# function broadcasted(::DefaultArrayStyle, ::typeof(*), a::Zeros, b::AbstractRange)
#     return Zeros{promote_type(eltype(a), eltype(b))}(broadcast_shape(size(a), size(b)))
# end

# function broadcasted(::DefaultArrayStyle, ::typeof(*), a::AbstractRange, b::Zeros)
#     return Zeros{promote_type(eltype(a), eltype(b))}(broadcast_shape(size(a), size(b)))
# end

function broadcasted(::DefaultArrayStyle, ::typeof(*), a::AbstractFill, b::AbstractRange)
    broadcast_shape(size(a), size(b)) # Check sizes are compatible.
    return broadcasted(*, getindex_value(a), b)
end

function broadcasted(::DefaultArrayStyle, ::typeof(*), a::AbstractRange, b::AbstractFill)
    broadcast_shape(size(a), size(b)) # Check sizes are compatible.
    return broadcasted(*, a, getindex_value(b))
end

broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}) where {T,N} = Fill(op(getindex_value(r)), size(r))
broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}, x::Number) where {T,N} = Fill(op(getindex_value(r),x), size(r))
broadcasted(::DefaultArrayStyle{N}, op, x::Number, r::AbstractFill{T,N}) where {T,N} = Fill(op(x, getindex_value(r)), size(r))
broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}, x::Ref) where {T,N} = Fill(op(getindex_value(r),x[]), size(r))
broadcasted(::DefaultArrayStyle{N}, op, x::Ref, r::AbstractFill{T,N}) where {T,N} = Fill(op(x[], getindex_value(r)), size(r))
# function broadcasted(::DefaultArrayStyle{N}, op, r1::AbstractFill{T,N}, r2::AbstractFill{V,N}) where {T,V,N}
#     size(r1) ≠ size(r2) && throw(DimensionMismatch("dimensions must match."))
#     Fill(op(getindex_value(r1),getindex_value(r2)), size(r1))
# end

# for op in (:*, :/, :\)
#     @eval function broadcasted(::DefaultArrayStyle{N}, ::typeof($op), r1::Ones{T,N}, r2::Ones{V,N}) where {T,V,N}
#         size(r1) ≠ size(r2) && throw(DimensionMismatch("dimensions must match."))
#         Ones{promote_type(T,V)}(size(r1))
#     end
# end
