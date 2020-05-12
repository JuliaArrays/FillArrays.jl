### map

map(f::Function, r::AbstractFill) = Fill(f(getindex_value(r)), axes(r))


### Unary broadcasting

function broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}) where {T,N}
    return Fill(op(getindex_value(r)), size(r))
end

broadcasted(::DefaultArrayStyle{N}, ::typeof(conj), r::Zeros{T,N}) where {T,N} = r
broadcasted(::DefaultArrayStyle{N}, ::typeof(conj), r::Ones{T,N}) where {T,N} = r
broadcasted(::DefaultArrayStyle{N}, ::typeof(real), r::Zeros{T,N}) where {T,N} = Zeros{real(T)}(r.axes)
broadcasted(::DefaultArrayStyle{N}, ::typeof(real), r::Ones{T,N}) where {T,N} = Ones{real(T)}(r.axes)
broadcasted(::DefaultArrayStyle{N}, ::typeof(imag), r::Zeros{T,N}) where {T,N} = Zeros{real(T)}(r.axes)
broadcasted(::DefaultArrayStyle{N}, ::typeof(imag), r::Ones{T,N}) where {T,N} = Zeros{real(T)}(r.axes)

### Binary broadcasting

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

broadcasted(::DefaultArrayStyle, ::typeof(*), a::Zeros, b::Zeros) = _broadcasted_zeros(a, b)

for op in (:*, :/)
    @eval begin
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Zeros, b::Fill) = _broadcasted_zeros(a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Zeros, b::Number) = _broadcasted_zeros(a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Zeros, b::AbstractRange) = _broadcasted_zeros(a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Zeros, b::AbstractArray) = _broadcasted_zeros(a, b)
        broadcasted(::DefaultArrayStyle{1}, ::typeof($op), a::Zeros, b::AbstractRange) = _broadcasted_zeros(a, b)
    end
end

for op in (:*, :\)
    @eval begin
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Fill, b::Zeros) = _broadcasted_zeros(a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Number, b::Zeros) = _broadcasted_zeros(a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractRange, b::Zeros) = _broadcasted_zeros(a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractArray, b::Zeros) = _broadcasted_zeros(a, b)
        broadcasted(::DefaultArrayStyle{1}, ::typeof($op), a::AbstractRange, b::Zeros) = _broadcasted_zeros(a, b)
    end
end

# special case due to missing converts for ranges
_range_convert(::Type{AbstractVector{T}}, a::AbstractRange{T}) where T = a
_range_convert(::Type{AbstractVector{T}}, a::AbstractUnitRange) where T = convert(T,first(a)):convert(T,last(a))
_range_convert(::Type{AbstractVector{T}}, a::AbstractRange) where T = convert(T,first(a)):step(a):convert(T,last(a))

function broadcasted(::DefaultArrayStyle{1}, ::typeof(*), a::Ones{T}, b::AbstractRange{V}) where {T,V}
    broadcast_shape(size(a), size(b)) # Check sizes are compatible.
    return _range_convert(AbstractVector{promote_type(T,V)}, b)
end

function broadcasted(::DefaultArrayStyle{1}, ::typeof(*), a::AbstractRange{V}, b::Ones{T}) where {T,V}
    broadcast_shape(size(a), size(b)) # Check sizes are compatible.
    return _range_convert(AbstractVector{promote_type(T,V)}, a)
end


function broadcasted(::DefaultArrayStyle{1}, ::typeof(*), a::AbstractFill, b::AbstractRange)
    broadcast_shape(size(a), size(b)) # Check sizes are compatible.
    return broadcasted(*, getindex_value(a), b)
end

function broadcasted(::DefaultArrayStyle{1}, ::typeof(*), a::AbstractRange, b::AbstractFill)
    broadcast_shape(size(a), size(b)) # Check sizes are compatible.
    return broadcasted(*, a, getindex_value(b))
end

broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}, x::Number) where {T,N} = Fill(op(getindex_value(r),x), size(r))
broadcasted(::DefaultArrayStyle{N}, op, x::Number, r::AbstractFill{T,N}) where {T,N} = Fill(op(x, getindex_value(r)), size(r))
broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}, x::Ref) where {T,N} = Fill(op(getindex_value(r),x[]), size(r))
broadcasted(::DefaultArrayStyle{N}, op, x::Ref, r::AbstractFill{T,N}) where {T,N} = Fill(op(x[], getindex_value(r)), size(r))


# Fast paths (no-op) when 2nd arg is Zeros or Ones
for (op, B) in [(:+, Zeros), 
                (:-, Zeros),
                (:*, Ones),
                (:/, Ones)]

    @eval function broadcasted(::DefaultArrayStyle, ::typeof($op), 
                            a::AbstractArray, b::$B)
        bs = broadcast_shape(size(a), size(b))
        Tout = promote_type(eltype(a), eltype(b))
        if size(a) == bs && eltype(a) == Tout
            return a
        end
        c = similar(a, Tout, bs)
        c .= a
        return c
    end

    for A in [Zeros, Ones]
        @eval function broadcasted(::DefaultArrayStyle, ::typeof($op), 
                            a::$A, b::$B)
            bs = broadcast_shape(size(a), size(b))
            Tout = promote_type(eltype(a), eltype(b))
            if size(a) == bs && eltype(a) == Tout
                return a
            end
            return $A{Tout}(bs)
        end
    end

    @eval function broadcasted(::DefaultArrayStyle, ::typeof($op), 
                            a::Fill, b::$B)
        bs = broadcast_shape(size(a), size(b))
        Tout = promote_type(eltype(a), eltype(b))
        if size(a) == bs && eltype(a) == Tout
            return a
        end
        return Fill{Tout}(getindex_value(a), bs)
    end
end


# Fast paths (no-op) when 1st arg is Zeros or Ones
for (op, A) in [(:+, Zeros), 
                (:*, Ones),
                (:\, Ones)] 

    @eval function broadcasted(::DefaultArrayStyle, ::typeof($op), a::$A, b::AbstractArray)
        bs = broadcast_shape(size(a), size(b))
        Tout = promote_type(eltype(a), eltype(b))
        if size(b) == bs && eltype(b) == Tout
            return b
        end
        c = similar(b, Tout, bs)
        c .= b
        return c
    end

    for B in [Zeros, Ones]
        if B != A  || op ∈ [:\]  # else defined when dealing with 2nd arg
            @eval function broadcasted(::DefaultArrayStyle, ::typeof($op), a::$A, b::$B)
                bs = broadcast_shape(size(a), size(b))
                Tout = promote_type(eltype(a), eltype(b))
                if size(b) == bs && eltype(b) == Tout
                    return b
                end
                return $B{Tout}(bs)
            end
        end
    end

    @eval function broadcasted(::DefaultArrayStyle, ::typeof($op), a::$A, b::Fill)
        bs = broadcast_shape(size(a), size(b))
        Tout = promote_type(eltype(a), eltype(b))
        if size(b) == bs && eltype(b) == Tout
            return b
        end
        return Fill{Tout}(getindex_value(b),  bs)
    end
end