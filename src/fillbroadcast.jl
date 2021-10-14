### map

map(f::Function, r::AbstractFill) = Fill(f(getindex_value(r)), axes(r))

function map(f::Function, vs::AbstractFill{<:Any,1}...)
    stop = mapreduce(length, min, vs)
    val = f(map(getindex_value, vs)...)
    Fill(val, stop)
end

function map(f::Function, rs::AbstractFill...)
    if _maplinear(rs...)
        map(f, map(vec, rs)...)
    else
        val = f(map(getindex_value, rs)...)
        Fill(val, axes(first(rs)))
    end
end

function _maplinear(rs...) # tries to match Base's behaviour, could perhaps hook in more deeply
    if any(ndims(r)==1 for r in rs)
        return true
    else
        r1 = axes(first(rs))
        for r in rs
            axes(r) == r1 || throw(DimensionMismatch(
            "dimensions must match: a has dims $r1, b has dims $(axes(r))"))
        end
        return false
    end
end

### mapreduce

if VERSION >= v"1.4"
    # _InitialValue was introduced after 1.0, before 1.4, not sure exact version.
    # Without these methods, some reductions will give an Array not a Fill.

    function Base._mapreduce_dim(f, op, ::Base._InitialValue, A::AbstractFill, ::Colon)
        fval = f(getindex_value(A))
        out = fval
        for _ in 2:length(A)
            out = op(out, fval)
        end
        out
    end

    function Base._mapreduce_dim(f, op, ::Base._InitialValue, A::AbstractFill, dims)
        fval = f(getindex_value(A))
        red = *(ntuple(d -> d in dims ? size(A,d) : 1, ndims(A))...)
        out = fval
        for _ in 2:red
            out = op(out, fval)
        end
        Fill(out, ntuple(d -> d in dims ? Base.OneTo(1) : axes(A,d), ndims(A)))
    end

end
if VERSION >= v"1.2" # Vararg mapreduce was added in Julia 1.2

    function mapreduce(f, op, A::AbstractFill, B::AbstractFill; kw...)
        val(_...) = f(getindex_value(A), getindex_value(B))
        reduce(op, map(val, A, B); kw...)
    end

    # These are particularly useful because mapreduce(*, +, A, B; dims) is slow in Base,
    # but can be re-written as some mapreduce(g, +, C; dims) which is fast.

    function mapreduce(f, op, A::AbstractFill, B::AbstractArray, Cs::AbstractArray...; kw...)
        g(b, cs...) = f(getindex_value(A), b, cs...)
        mapreduce(g, op, B, Cs...; kw...)
    end
    function mapreduce(f, op, A::AbstractArray, B::AbstractFill, Cs::AbstractArray...; kw...)
        h(a, cs...) = f(a, getindex_value(B), cs...)
        mapreduce(h, op, A, Cs...; kw...)
    end
    function mapreduce(f, op, A::AbstractFill, B::AbstractFill, Cs::AbstractArray...; kw...)
        gh(cs...) = f(getindex_value(A), getindex_value(B), cs...)
        mapreduce(gh, op, Cs...; kw...)
    end

end

### Unary broadcasting

function broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}) where {T,N}
    return Fill(op(getindex_value(r)), axes(r))
end

broadcasted(::DefaultArrayStyle, ::typeof(+), r::Zeros) = r
broadcasted(::DefaultArrayStyle, ::typeof(-), r::Zeros) = r
broadcasted(::DefaultArrayStyle, ::typeof(+), r::Ones) = r

broadcasted(::DefaultArrayStyle{N}, ::typeof(conj), r::Zeros{T,N}) where {T,N} = r
broadcasted(::DefaultArrayStyle{N}, ::typeof(conj), r::Ones{T,N}) where {T,N} = r
broadcasted(::DefaultArrayStyle{N}, ::typeof(real), r::Zeros{T,N}) where {T,N} = Zeros{real(T)}(r.axes)
broadcasted(::DefaultArrayStyle{N}, ::typeof(real), r::Ones{T,N}) where {T,N} = Ones{real(T)}(r.axes)
broadcasted(::DefaultArrayStyle{N}, ::typeof(imag), r::Zeros{T,N}) where {T,N} = Zeros{real(T)}(r.axes)
broadcasted(::DefaultArrayStyle{N}, ::typeof(imag), r::Ones{T,N}) where {T,N} = Zeros{real(T)}(r.axes)

### Binary broadcasting

function broadcasted(::DefaultArrayStyle, op, a::AbstractFill, b::AbstractFill)
    val = op(getindex_value(a), getindex_value(b))
    return Fill(val, broadcast_shape(axes(a), axes(b)))
end


_broadcasted_zeros(f, a, b) = Zeros{Base.Broadcast.combine_eltypes(f, (a, b))}(broadcast_shape(axes(a), axes(b)))
_broadcasted_ones(f, a, b) = Ones{Base.Broadcast.combine_eltypes(f, (a, b))}(broadcast_shape(axes(a), axes(b)))
_broadcasted_nan(f, a, b) = Fill(convert(Base.Broadcast.combine_eltypes(f, (a, b)), NaN), broadcast_shape(axes(a), axes(b)))

# TODO: remove at next breaking version
_broadcasted_zeros(a, b) = _broadcasted_zeros(+, a, b)
_broadcasted_ones(a, b) = _broadcasted_ones(+, a, b)

broadcasted(::DefaultArrayStyle, ::typeof(+), a::Zeros, b::Zeros) = _broadcasted_zeros(+, a, b)
broadcasted(::DefaultArrayStyle, ::typeof(+), a::Ones, b::Zeros) = _broadcasted_ones(+, a, b)
broadcasted(::DefaultArrayStyle, ::typeof(+), a::Zeros, b::Ones) = _broadcasted_ones(+, a, b)

broadcasted(::DefaultArrayStyle, ::typeof(-), a::Zeros, b::Zeros) = _broadcasted_zeros(-, a, b)
broadcasted(::DefaultArrayStyle, ::typeof(-), a::Ones, b::Zeros) = _broadcasted_ones(-, a, b)
broadcasted(::DefaultArrayStyle, ::typeof(-), a::Ones, b::Ones) = _broadcasted_zeros(-, a, b)

broadcasted(::DefaultArrayStyle{1}, ::typeof(+), a::Zeros{<:Any,1}, b::Zeros{<:Any,1}) = _broadcasted_zeros(+, a, b)
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), a::Ones{<:Any,1}, b::Zeros{<:Any,1}) = _broadcasted_ones(+, a, b)
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), a::Zeros{<:Any,1}, b::Ones{<:Any,1}) = _broadcasted_ones(+, a, b)

broadcasted(::DefaultArrayStyle{1}, ::typeof(-), a::Zeros{<:Any,1}, b::Zeros{<:Any,1}) = _broadcasted_zeros(-, a, b)
broadcasted(::DefaultArrayStyle{1}, ::typeof(-), a::Ones{<:Any,1}, b::Zeros{<:Any,1}) = _broadcasted_ones(-, a, b)


broadcasted(::DefaultArrayStyle, ::typeof(*), a::Zeros, b::Zeros) = _broadcasted_zeros(*, a, b)

# In following, need to restrict to <: Number as otherwise we cannot infer zero from type
# TODO: generalise to things like SVector
for op in (:*, :/)
    @eval begin
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Zeros, b::Ones) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Zeros, b::Fill{<:Number}) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Zeros, b::Number) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Zeros, b::AbstractRange) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Zeros, b::AbstractArray{<:Number}) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Zeros, b::Base.Broadcast.Broadcasted) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle{1}, ::typeof($op), a::Zeros, b::AbstractRange) = _broadcasted_zeros($op, a, b)
    end
end

for op in (:*, :\)
    @eval begin
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Ones, b::Zeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Fill{<:Number}, b::Zeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Number, b::Zeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractRange, b::Zeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractArray{<:Number}, b::Zeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Base.Broadcast.Broadcasted, b::Zeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle{1}, ::typeof($op), a::AbstractRange, b::Zeros) = _broadcasted_zeros($op, a, b)
    end
end

for op in (:*, :/, :\)
    @eval broadcasted(::DefaultArrayStyle, ::typeof($op), a::Ones, b::Ones) = _broadcasted_ones($op, a, b)
end

for op in (:/, :\)
    @eval broadcasted(::DefaultArrayStyle, ::typeof($op), a::Zeros{<:Number}, b::Zeros{<:Number}) = _broadcasted_nan($op, a, b)
end

# special case due to missing converts for ranges
_range_convert(::Type{AbstractVector{T}}, a::AbstractRange{T}) where T = a
_range_convert(::Type{AbstractVector{T}}, a::AbstractUnitRange) where T = convert(T,first(a)):convert(T,last(a))
_range_convert(::Type{AbstractVector{T}}, a::AbstractRange) where T = convert(T,first(a)):step(a):convert(T,last(a))


# TODO: replacing with the following will support more general broadcasting.
# function broadcasted(::DefaultArrayStyle{1}, ::typeof(*), a::AbstractFill, b::AbstractRange)
#     broadcast_shape(axes(a), axes(b)) # check axes
#     r1 = b[1] * getindex_value(a)
#     T = typeof(r1)
#     if length(b) == 1 # Need a fill, but for type stability use StepRangeLen
#         StepRangeLen{T}(r1, zero(T), length(a))
#     else
#         StepRangeLen{T}(r1, convert(T, getindex_value(a) * step(b)), length(b))
#     end
# end

# function broadcasted(::DefaultArrayStyle{1}, ::typeof(*), a::AbstractRange, b::AbstractFill)
#     broadcast_shape(axes(a), axes(b)) # check axes
#     r1 = a[1] * getindex_value(b)
#     T = typeof(r1)
#     if length(a) == 1 # Need a fill, but for type stability use StepRangeLen
#         StepRangeLen{T}(r1, zero(T), length(b))
#     else
#         StepRangeLen{T}(r1, convert(T, step(a) * getindex_value(b)), length(a))
#     end
# end

function broadcasted(::DefaultArrayStyle{1}, ::typeof(*), a::Ones{T,1}, b::AbstractRange{V}) where {T,V}
    broadcast_shape(axes(a), axes(b)) == axes(b) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $b to a Vector first."))
    return _range_convert(AbstractVector{promote_type(T,V)}, b)
end

function broadcasted(::DefaultArrayStyle{1}, ::typeof(*), a::AbstractRange{V}, b::Ones{T,1}) where {T,V}
    broadcast_shape(axes(a), axes(b)) == axes(a) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $b to a Vector first."))
    return _range_convert(AbstractVector{promote_type(T,V)}, a)
end

for op in (:+, -)
    @eval begin
        function broadcasted(::DefaultArrayStyle{1}, ::typeof($op), a::AbstractVector{T}, b::Zeros{V,1}) where {T,V}
            broadcast_shape(axes(a), axes(b)) == axes(a) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $b to a Vector first."))
            LinearAlgebra.copy_oftype(a, promote_type(T,V))
        end

        broadcasted(::DefaultArrayStyle{1}, ::typeof($op), a::AbstractFill{T,1}, b::Zeros{V,1}) where {T,V} =
            Base.invoke(broadcasted, Tuple{DefaultArrayStyle, typeof($op), AbstractFill, AbstractFill}, DefaultArrayStyle{1}(), $op, a, b)
    end
end

function broadcasted(::DefaultArrayStyle{1}, ::typeof(+), a::Zeros{T,1}, b::AbstractVector{V}) where {T,V}
    broadcast_shape(axes(a), axes(b))
    LinearAlgebra.copy_oftype(b, promote_type(T,V))
end

broadcasted(::DefaultArrayStyle{1}, ::typeof(+), a::Zeros{V,1}, b::AbstractFill{T,1}) where {T,V} =
            Base.invoke(broadcasted, Tuple{DefaultArrayStyle, typeof(+), AbstractFill, AbstractFill}, DefaultArrayStyle{1}(), +, a, b)

# Need to prevent array-valued fills from broadcasting over entry
_broadcast_getindex_value(a::AbstractFill{<:Number}) = getindex_value(a)
_broadcast_getindex_value(a::AbstractFill) = Ref(getindex_value(a))


function broadcasted(::DefaultArrayStyle{1}, ::typeof(*), a::AbstractFill, b::AbstractRange)
    broadcast_shape(axes(a), axes(b)) == axes(b) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $b to a Vector first."))
    return broadcasted(*, _broadcast_getindex_value(a), b)
end

function broadcasted(::DefaultArrayStyle{1}, ::typeof(*), a::AbstractRange, b::AbstractFill)
    broadcast_shape(axes(a), axes(b)) == axes(a) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $b to a Vector first."))
    return broadcasted(*, a, _broadcast_getindex_value(b))
end

broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}, x::Number) where {T,N} = Fill(op(getindex_value(r),x), axes(r))
broadcasted(::DefaultArrayStyle{N}, op, x::Number, r::AbstractFill{T,N}) where {T,N} = Fill(op(x, getindex_value(r)), axes(r))
broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}, x::Ref) where {T,N} = Fill(op(getindex_value(r),x[]), axes(r))
broadcasted(::DefaultArrayStyle{N}, op, x::Ref, r::AbstractFill{T,N}) where {T,N} = Fill(op(x[], getindex_value(r)), axes(r))

# support AbstractFill .^ k
broadcasted(::DefaultArrayStyle{N}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, r::AbstractFill{T,N}, ::Base.RefValue{Val{k}}) where {T,N,k} = Fill(getindex_value(r)^k, axes(r))
broadcasted(::DefaultArrayStyle{N}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, r::Ones{T,N}, ::Base.RefValue{Val{k}}) where {T,N,k} = Ones{T}(axes(r))
broadcasted(::DefaultArrayStyle{N}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, r::Zeros{T,N}, ::Base.RefValue{Val{0}}) where {T,N} = Ones{T}(axes(r))
broadcasted(::DefaultArrayStyle{N}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, r::Zeros{T,N}, ::Base.RefValue{Val{k}}) where {T,N,k} = Zeros{T}(axes(r))

# Special broadcasting rules involving structured matrices
const DiagonalFill = Diagonal{<:Any, <:AbstractFill}
broadcasted(::StructuredMatrixStyle, ::typeof(+), D1::DiagonalFill, D2::DiagonalFill) = Diagonal(parent(D1) .+ parent(D2))
broadcasted(::StructuredMatrixStyle, ::typeof(-), D1::DiagonalFill, D2::DiagonalFill) = Diagonal(parent(D1) .- parent(D2))

broadcasted(::StructuredMatrixStyle, ::typeof(*), D::DiagonalFill, x::Number) = Diagonal(parent(D) .* x)
broadcasted(::StructuredMatrixStyle, ::typeof(*), x::Number, D::DiagonalFill) = Diagonal(x .* parent(D))
broadcasted(::StructuredMatrixStyle, ::typeof(*), D1::DiagonalFill, D2::DiagonalFill) = Diagonal(parent(D1) .* parent(D2))

broadcasted(::StructuredMatrixStyle, ::typeof(/), D::DiagonalFill, x::Number) = Diagonal(parent(D) ./ x)
broadcasted(::StructuredMatrixStyle, ::typeof(\), x::Number, D::DiagonalFill) = Diagonal(x .\ parent(D))

broadcasted(::StructuredMatrixStyle, ::typeof(*), D::DiagonalFill, x::AbstractRange) = Diagonal(parent(D) .* x)
broadcasted(::StructuredMatrixStyle, ::typeof(*), x::AbstractRange, D::DiagonalFill) = Diagonal(x .* parent(D))

broadcasted(::StructuredMatrixStyle, ::typeof(^), D::DiagonalFill, x::Number) = Diagonal(parent(D) .^ x)

broadcasted(::StructuredMatrixStyle, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, D::DiagonalFill, ::Base.RefValue{Val{k}}) where {k} = Diagonal(parent(D) .^ k)
