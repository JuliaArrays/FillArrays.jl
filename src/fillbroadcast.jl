for op in (:+, :-)
    @eval broadcasted(::DefaultArrayStyle{N}, ::typeof($op), r1::AbstractFill{T,N}, r2::AbstractFill{V,N}) where {T,V,N} =
            $op(r1, r2)
end

function broadcasted(::DefaultArrayStyle{N}, ::typeof(*), a::Zeros{T,N}, b::Zeros{V,N}) where {T,V,N}
    axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
    Zeros{promote_type(T,V)}(axes(a))
end

function _broadcasted_mul(a::AbstractArray{T}, b::Zeros{V}) where {T,V}
    axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
    Zeros{promote_type(T,V)}(axes(a))
end
function broadcasted(::DefaultArrayStyle{N}, ::typeof(*), a::AbstractArray{T,N}, b::Zeros{V,N}) where {T,V,N}
    return _broadcasted_mul(a, b)
end
function broadcasted(::DefaultArrayStyle{N}, ::typeof(*), a::AbstractFill{T,N}, b::Zeros{V,N}) where {T,V,N}
    return _broadcasted_mul(a, b)
end

function _broadcasted_mul(a::Zeros{T}, b::AbstractArray{V}) where {T,V}
    axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
    Zeros{promote_type(T,V)}(axes(a))
end
function broadcasted(::DefaultArrayStyle{N}, ::typeof(*), a::Zeros{T,N}, b::AbstractArray{V,N}) where {T,V,N}
    _broadcasted_mul(a, b)
end
function broadcasted(::DefaultArrayStyle{N}, ::typeof(*), a::Zeros{T,N}, b::AbstractFill{V,N}) where {T,V,N}
    _broadcasted_mul(a, b)
end

function broadcasted(::DefaultArrayStyle, ::typeof(*), a::Zeros, b::AbstractRange)
    return Zeros{promote_type(eltype(a), eltype(b))}(broadcast_shape(size(a), size(b)))
end

function broadcasted(::DefaultArrayStyle, ::typeof(*), a::AbstractRange, b::Zeros)
    return Zeros{promote_type(eltype(a), eltype(b))}(broadcast_shape(size(a), size(b)))
end

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
