### map

map(f::Function, r::AbstractFill) = Fill(f(getindex_value(r)), axes(r))

function map(f::Function, v::AbstractFillVector, ws::AbstractFillVector...)
    stop = mapreduce(length, min, (v, ws...))
    val = f(map(getindex_value, (v, ws...))...)
    Fill(val, stop)
end

function map(f::Function, q::AbstractFill, rs::AbstractFill...)
    if _maplinear(q, rs...)
        map(f, map(vec, (q, rs...))...)
    else
        val = f(map(getindex_value, (q, rs...))...)
        Fill(val, axes(q))
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

function Base._mapreduce_dim(f, op, ::Base._InitialValue, A::AbstractFill, ::Colon)
    if length(A) == 0
        return Base.mapreduce_empty_iter(f, op, A, Base.HasEltype())
    end
    val = getindex_value(A)
    out = Base.mapreduce_first(f, op, val)
    fval = f(val)
    if op(out, fval) != out
        for _ in 2:length(A)
            out = op(out, fval)
        end
    end
    out
end

function Base._mapreduce_dim(f, op, init, A::AbstractFill, ::Colon)
    if length(A) == 0
        return init
    end
    val = getindex_value(A)
    fval = f(val)
    out = op(init, fval)
    if op(out, fval) != out
        for _ in 2:length(A)
            out = op(out, fval)
        end
    end
    out
end

identityel(f, ::Union{typeof(+), typeof(Base.add_sum)}, A) = zero(f(zero(eltype(A))))
identityel(f, ::Union{typeof(*), typeof(Base.mul_prod)}, A) = one(f(one(eltype(A))))
identityel(f, ::typeof(&), A) = true
identityel(f, ::typeof(|), A) = false
identityel(f, ::Any, @nospecialize(A)) = throw(ArgumentError("reducing over an empty collection is not allowed"))
function mapreducedim_empty(f, op, A)
    z = identityel(f, op, A)
    op(z, z)
end

function reduced_indices(A, dims)
    ntuple(d -> d in dims ? axes(A,ndims(A)+1) : axes(A,d), ndims(A))
end

function Base._mapreduce_dim(f, op, ::Base._InitialValue, A::AbstractFill, dims)
    red = *(ntuple(d -> d in dims ? size(A,d) : 1, ndims(A))...)
    if red == 0
        out = mapreducedim_empty(f, op, A)
    else
        val = getindex_value(A)
        out = Base.mapreduce_first(f, op, val)
        fval = f(val)
        if op(out, fval) != out
            for _ in 2:red
                out = op(out, fval)
            end
        end
    end
    Fill(out, reduced_indices(A, dims))
end

function Base._mapreduce_dim(f, op, init, A::AbstractFill, dims)
    red = *(ntuple(d -> d in dims ? size(A,d) : 1, ndims(A))...)
    if red == 0
        out = init
    else
        val = getindex_value(A)
        fval = f(val)
        out = op(init, fval)
        if op(out, fval) != out
            for _ in 2:red
                out = op(out, fval)
            end
        end
    end
    Fill(out, reduced_indices(A, dims))
end

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


### Unary broadcasting

function broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}) where {T,N}
    return Fill(op(getindex_value(r)), axes(r))
end

broadcasted(::DefaultArrayStyle, ::typeof(+), r::AbstractZeros) = r
broadcasted(::DefaultArrayStyle, ::typeof(-), r::AbstractZeros) = r
broadcasted(::DefaultArrayStyle, ::typeof(+), r::AbstractOnes) = r

broadcasted(::DefaultArrayStyle{N}, ::typeof(conj), r::AbstractZeros{T,N}) where {T,N} = r
broadcasted(::DefaultArrayStyle{N}, ::typeof(conj), r::AbstractOnes{T,N}) where {T,N} = r
broadcasted(::DefaultArrayStyle{N}, ::typeof(real), r::AbstractZeros{T,N}) where {T,N} = Zeros{real(T)}(r.axes)
broadcasted(::DefaultArrayStyle{N}, ::typeof(real), r::AbstractOnes{T,N}) where {T,N} = Ones{real(T)}(r.axes)
broadcasted(::DefaultArrayStyle{N}, ::typeof(imag), r::AbstractZeros{T,N}) where {T,N} = Zeros{real(T)}(r.axes)
broadcasted(::DefaultArrayStyle{N}, ::typeof(imag), r::AbstractOnes{T,N}) where {T,N} = Zeros{real(T)}(r.axes)

### Binary broadcasting

# Default outputs, can overload to customize
broadcasted_fill(f, a, val, ax) = Fill(val, ax)
broadcasted_fill(f, a, b, val, ax) = Fill(val, ax)
broadcasted_zeros(f, a, elt, ax) = Zeros{elt}(ax)
broadcasted_zeros(f, a, b, elt, ax) = Zeros{elt}(ax)
broadcasted_ones(f, a, elt, ax) = Ones{elt}(ax)
broadcasted_ones(f, a, b, elt, ax) = Ones{elt}(ax)

function broadcasted(::DefaultArrayStyle, op, a::AbstractFill, b::AbstractFill)
    val = op(getindex_value(a), getindex_value(b))
    ax = broadcast_shape(axes(a), axes(b))
    return broadcasted_fill(op, a, b, val, ax)
end

function _broadcasted_zeros(f, a, b)
  elt = Base.Broadcast.combine_eltypes(f, (a, b))
  ax = broadcast_shape(axes(a), axes(b))
  return broadcasted_zeros(f, a, b, elt, ax)
end
function _broadcasted_ones(f, a, b)
  elt = Base.Broadcast.combine_eltypes(f, (a, b))
  ax = broadcast_shape(axes(a), axes(b))
  return broadcasted_ones(f, a, b, elt, ax)
end
function _broadcasted_nan(f, a, b)
  val = convert(Base.Broadcast.combine_eltypes(f, (a, b)), NaN)
  ax = broadcast_shape(axes(a), axes(b))
  return broadcasted_fill(f, a, b, val, ax)
end

broadcasted(::DefaultArrayStyle, ::typeof(+), a::AbstractZeros, b::AbstractZeros) = _broadcasted_zeros(+, a, b)
broadcasted(::DefaultArrayStyle, ::typeof(+), a::AbstractOnes, b::AbstractZeros) = _broadcasted_ones(+, a, b)
broadcasted(::DefaultArrayStyle, ::typeof(+), a::AbstractZeros, b::AbstractOnes) = _broadcasted_ones(+, a, b)

broadcasted(::DefaultArrayStyle, ::typeof(-), a::AbstractZeros, b::AbstractZeros) = _broadcasted_zeros(-, a, b)
broadcasted(::DefaultArrayStyle, ::typeof(-), a::AbstractOnes, b::AbstractZeros) = _broadcasted_ones(-, a, b)
broadcasted(::DefaultArrayStyle, ::typeof(-), a::AbstractOnes, b::AbstractOnes) = _broadcasted_zeros(-, a, b)

broadcasted(::DefaultArrayStyle{1}, ::typeof(+), a::AbstractZerosVector, b::AbstractZerosVector) = _broadcasted_zeros(+, a, b)
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), a::AbstractOnesVector, b::AbstractZerosVector) = _broadcasted_ones(+, a, b)
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), a::AbstractZerosVector, b::AbstractOnesVector) = _broadcasted_ones(+, a, b)

broadcasted(::DefaultArrayStyle{1}, ::typeof(-), a::AbstractZerosVector, b::AbstractZerosVector) = _broadcasted_zeros(-, a, b)
broadcasted(::DefaultArrayStyle{1}, ::typeof(-), a::AbstractOnesVector, b::AbstractZerosVector) = _broadcasted_ones(-, a, b)


broadcasted(::DefaultArrayStyle, ::typeof(*), a::AbstractZeros, b::AbstractZeros) = _broadcasted_zeros(*, a, b)

# In following, need to restrict to <: Number as otherwise we cannot infer zero from type
# TODO: generalise to things like SVector
for op in (:*, :/)
    @eval begin
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractZeros, b::AbstractOnes) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractZeros, b::Fill{<:Number}) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractZeros, b::Number) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractZeros, b::AbstractRange) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractZeros, b::AbstractArray{<:Number}) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractZeros, b::Base.Broadcast.Broadcasted) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle{1}, ::typeof($op), a::AbstractZeros, b::AbstractRange) = _broadcasted_zeros($op, a, b)
    end
end

for op in (:*, :\)
    @eval begin
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractOnes, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Fill{<:Number}, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Number, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractRange, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractArray{<:Number}, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle, ::typeof($op), a::Base.Broadcast.Broadcasted, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::DefaultArrayStyle{1}, ::typeof($op), a::AbstractRange, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
    end
end

for op in (:*, :/, :\)
    @eval broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractOnes, b::AbstractOnes) = _broadcasted_ones($op, a, b)
end

for op in (:/, :\)
    @eval broadcasted(::DefaultArrayStyle, ::typeof($op), a::AbstractZeros{<:Number}, b::AbstractZeros{<:Number}) = _broadcasted_nan($op, a, b)
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

function broadcasted(::DefaultArrayStyle{1}, ::typeof(*), a::AbstractOnesVector, b::AbstractRange)
    broadcast_shape(axes(a), axes(b)) == axes(b) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $b to a Vector first."))
    TT = typeof(zero(eltype(a)) * zero(eltype(b)))
    return _range_convert(AbstractVector{TT}, b)
end

function broadcasted(::DefaultArrayStyle{1}, ::typeof(*), a::AbstractRange, b::AbstractOnesVector)
    broadcast_shape(axes(a), axes(b)) == axes(a) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $b to a Vector first."))
    TT = typeof(zero(eltype(a)) * zero(eltype(b)))
    return _range_convert(AbstractVector{TT}, a)
end

for op in (:+, :-)
    @eval begin
        function broadcasted(::DefaultArrayStyle{1}, ::typeof($op), a::AbstractVector, b::AbstractZerosVector)
            broadcast_shape(axes(a), axes(b)) == axes(a) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $b to a Vector first."))
            TT = typeof($op(zero(eltype(a)), zero(eltype(b))))
            # Use `TT ∘ (+)` to fix AD issues with `broadcasted(TT, x)`
            eltype(a) === TT ? a : broadcasted(TT ∘ (+), a)
        end
        function broadcasted(::DefaultArrayStyle{1}, ::typeof($op), a::AbstractZerosVector, b::AbstractVector)
            broadcast_shape(axes(a), axes(b)) == axes(b) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $a to a Vector first."))
            TT = typeof($op(zero(eltype(a)), zero(eltype(b))))
            $op === (+) && eltype(b) === TT ? b : broadcasted(TT ∘ ($op), b)
        end

        broadcasted(::DefaultArrayStyle{1}, ::typeof($op), a::AbstractFillVector, b::AbstractZerosVector) =
            Base.invoke(broadcasted, Tuple{DefaultArrayStyle, typeof($op), AbstractFill, AbstractFill}, DefaultArrayStyle{1}(), $op, a, b)

        broadcasted(::DefaultArrayStyle{1}, ::typeof($op), a::AbstractZerosVector, b::AbstractFillVector) =
            Base.invoke(broadcasted, Tuple{DefaultArrayStyle, typeof($op), AbstractFill, AbstractFill}, DefaultArrayStyle{1}(), $op, a, b)
    end
end

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

broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}, x::Number) where {T,N} = broadcasted_fill(op, r, op(getindex_value(r),x), axes(r))
broadcasted(::DefaultArrayStyle{N}, op, x::Number, r::AbstractFill{T,N}) where {T,N} = broadcasted_fill(op, r, op(x, getindex_value(r)), axes(r))
broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill{T,N}, x::Ref) where {T,N} = broadcasted_fill(op, r, op(getindex_value(r),x[]), axes(r))
broadcasted(::DefaultArrayStyle{N}, op, x::Ref, r::AbstractFill{T,N}) where {T,N} = broadcasted_fill(op, r, op(x[], getindex_value(r)), axes(r))

# support AbstractFill .^ k
broadcasted(::DefaultArrayStyle{N}, op::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, r::AbstractFill{T,N}, ::Base.RefValue{Val{k}}) where {T,N,k} = broadcasted_fill(op, r, getindex_value(r)^k, axes(r))
broadcasted(::DefaultArrayStyle{N}, op::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, r::AbstractOnes{T,N}, ::Base.RefValue{Val{k}}) where {T,N,k} = broadcasted_ones(op, r, T, axes(r))
broadcasted(::DefaultArrayStyle{N}, op::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, r::AbstractZeros{T,N}, ::Base.RefValue{Val{0}}) where {T,N} = broadcasted_ones(op, r, T, axes(r))
broadcasted(::DefaultArrayStyle{N}, op::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, r::AbstractZeros{T,N}, ::Base.RefValue{Val{k}}) where {T,N,k} = broadcasted_zeros(op, r, T, axes(r))

# supports structured broadcast
if isdefined(LinearAlgebra, :fzero)
    LinearAlgebra.fzero(x::AbstractZeros) = zero(eltype(x))
end
