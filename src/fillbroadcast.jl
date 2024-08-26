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


## BroadcastStyle

abstract type AbstractFillStyle{N} <: Broadcast.AbstractArrayStyle{N} end
struct FillStyle{N} <: AbstractFillStyle{N} end
struct ZerosStyle{N} <: AbstractFillStyle{N} end
FillStyle{N}(::Val{M}) where {N,M} = FillStyle{M}()
ZerosStyle{N}(::Val{M}) where {N,M} = ZerosStyle{M}()
Broadcast.BroadcastStyle(::Type{<:AbstractFill{<:Any,N}}) where {N} = FillStyle{N}()
Broadcast.BroadcastStyle(::Type{<:AbstractZeros{<:Any,N}}) where {N} = ZerosStyle{N}()
Broadcast.BroadcastStyle(::FillStyle{M}, ::ZerosStyle{N}) where {M,N} = FillStyle{max(M,N)}()
Broadcast.BroadcastStyle(S::LinearAlgebra.StructuredMatrixStyle, ::ZerosStyle{2}) = S
Broadcast.BroadcastStyle(S::LinearAlgebra.StructuredMatrixStyle, ::ZerosStyle{1}) = S
Broadcast.BroadcastStyle(S::LinearAlgebra.StructuredMatrixStyle, ::ZerosStyle{0}) = S

_getindex_value(f::AbstractFill) = getindex_value(f)
_getindex_value(x::Number) = x
_getindex_value(x::Ref) = x[]
function _getindex_value(bc::Broadcast.Broadcasted)
    bc.f(map(_getindex_value, bc.args)...)
end

has_static_value(x) = false
has_static_value(x::Union{AbstractZeros, AbstractOnes}) = true
has_static_value(x::Broadcast.Broadcasted) = all(has_static_value, x.args)

function _iszeros(bc::Broadcast.Broadcasted)
    all(has_static_value, bc.args) && _iszero(_getindex_value(bc))
end
# conservative check for zeros. In most cases, there isn't a zero element to compare with
_iszero(x::Union{Number, AbstractArray}) = iszero(x)
_iszero(_) = false

function _isones(bc::Broadcast.Broadcasted)
    all(has_static_value, bc.args) && _isone(_getindex_value(bc))
end
# conservative check for ones. In most cases, there isn't a unit element to compare with
_isone(x::Union{Number, AbstractArray}) = isone(x)
_isone(_) = false

_isfill(bc::Broadcast.Broadcasted) = all(_isfill, bc.args)
_isfill(f::AbstractFill) = true
_isfill(f::Number) = true
_isfill(f::Ref) = true
_isfill(::Any) = false

function Base.copy(bc::Broadcast.Broadcasted{<:AbstractFillStyle{N}}) where {N}
    if _iszeros(bc)
        return Zeros(typeof(_getindex_value(bc)), axes(bc))
    elseif _isones(bc)
        return Ones(typeof(_getindex_value(bc)), axes(bc))
    elseif _isfill(bc)
        return Fill(_getindex_value(bc), axes(bc))
    else
        # fallback style
        S = Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{N}}
        copy(convert(S, bc))
    end
end
# make the zero-dimensional case consistent with Base
function Base.copy(bc::Broadcast.Broadcasted{<:AbstractFillStyle{0}})
    S = Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{0}}
    copy(convert(S, bc))
end

# some cases that preserve 0d
function broadcast_preserving_0d(f, As...)
    bc = Base.broadcasted(f, As...)
    r = copy(bc)
    length(axes(bc)) == 0 ? Fill(r) : r
end
for f in (:real, :imag)
    @eval ($f)(A::AbstractFill) = broadcast_preserving_0d($f, A)
    @eval ($f)(A::AbstractZeros) = Zeros{real(eltype(A))}(axes(A))
end
conj(A::AbstractFill) = broadcast_preserving_0d(conj, A)
conj(A::AbstractZeros) = A
real(A::AbstractOnes) = Ones{real(eltype(A))}(axes(A))
imag(A::AbstractOnes) = Zeros{real(eltype(A))}(axes(A))
conj(A::AbstractOnes) = A
real(A::AbstractFill{<:Real}) = A
imag(A::AbstractFill{<:Real}) = Zeros{eltype(A)}(axes(A))
conj(A::AbstractFill{<:Real}) = A

### Binary broadcasting

# Default outputs, can overload to customize
broadcasted_fill(f, a, val, ax) = Fill(val, ax)
broadcasted_fill(f, a, b, val, ax) = Fill(val, ax)
broadcasted_zeros(f, a, elt, ax) = Zeros{elt}(ax)
broadcasted_zeros(f, a, b, elt, ax) = Zeros{elt}(ax)
broadcasted_ones(f, a, elt, ax) = Ones{elt}(ax)
broadcasted_ones(f, a, b, elt, ax) = Ones{elt}(ax)

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

# In following, need to restrict to <: Number as otherwise we cannot infer zero from type
# TODO: generalise to things like SVector
for op in (:*, :/)
    @eval begin
        broadcasted(::typeof($op), a::AbstractZeros, b::AbstractFill{<:Number}) = _broadcasted_zeros($op, a, b)
        broadcasted(::typeof($op), a::AbstractZeros, b::Number) = _broadcasted_zeros($op, a, b)
        broadcasted(::typeof($op), a::AbstractZeros, b::AbstractOnes) = _broadcasted_zeros($op, a, b)
        broadcasted(::typeof($op), a::AbstractZeros, b::AbstractRange) = _broadcasted_zeros($op, a, b)
        broadcasted(::typeof($op), a::AbstractZeros, b::AbstractArray{<:Number}) = _broadcasted_zeros($op, a, b)
        broadcasted(::typeof($op), a::AbstractZeros, b::Base.Broadcast.Broadcasted) = _broadcasted_zeros($op, a, b)
    end
end

for op in (:*, :\)
    @eval begin
        broadcasted(::typeof($op), a::AbstractOnes, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::typeof($op), a::AbstractFill{<:Number}, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::typeof($op), a::Number, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::typeof($op), a::AbstractRange, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::typeof($op), a::AbstractArray{<:Number}, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::typeof($op), a::Base.Broadcast.Broadcasted, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
    end
end
broadcasted(::typeof(*), a::AbstractZeros, b::AbstractZeros) = _broadcasted_zeros(*, a, b)
broadcasted(::typeof(/), a::AbstractZeros, b::AbstractZeros) = _broadcasted_nan(/, a, b)
broadcasted(::typeof(\), a::AbstractZeros, b::AbstractZeros) = _broadcasted_nan(\, a, b)

# special case due to missing converts for ranges
_range_convert(::Type{AbstractVector{T}}, a::AbstractRange{T}) where T = a
_range_convert(::Type{AbstractVector{T}}, a::AbstractUnitRange) where T = convert(T,first(a)):convert(T,last(a))
_range_convert(::Type{AbstractVector{T}}, a::OneTo) where T = OneTo(convert(T, a.stop))
_range_convert(::Type{AbstractVector{T}}, a::AbstractRange) where T = convert(T,first(a)):step(a):convert(T,last(a))
_range_convert(::Type{AbstractVector{T}}, a::ZerosVector) where T = ZerosVector{T}(length(a))


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

function broadcasted(::FillStyle{1}, ::typeof(*), a::AbstractOnes, b::AbstractRange)
    broadcast_shape(axes(a), axes(b)) == axes(b) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $b to a Vector first."))
    TT = typeof(zero(eltype(a)) * zero(eltype(b)))
    return _range_convert(AbstractVector{TT}, b)
end

function broadcasted(::FillStyle{1}, ::typeof(*), a::AbstractRange, b::AbstractOnes)
    broadcast_shape(axes(a), axes(b)) == axes(a) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $b to a Vector first."))
    TT = typeof(zero(eltype(a)) * zero(eltype(b)))
    return _range_convert(AbstractVector{TT}, a)
end

for op in (:+, :-)
    @eval begin
        function broadcasted(::typeof($op), a::AbstractVector, b::AbstractZerosVector)
            ax = broadcast_shape(axes(a), axes(b))
            ax == axes(a) || throw(ArgumentError("cannot broadcast an array with size $(size(a)) with $b"))
            TT = typeof($op(zero(eltype(a)), zero(eltype(b))))
            # Use `TT ∘ (+)` to fix AD issues with `broadcasted(TT, x)`
            eltype(a) === TT ? a : broadcasted(TT ∘ (+), a)
        end
        function broadcasted(::typeof($op), a::AbstractZerosVector, b::AbstractVector)
            ax = broadcast_shape(axes(a), axes(b))
            ax == axes(b) || throw(ArgumentError("cannot broadcast $a with an array with size $(size(b))"))
            TT = typeof($op(zero(eltype(a)), zero(eltype(b))))
            $op === (+) && eltype(b) === TT ? b : broadcasted(TT ∘ ($op), b)
        end
        function broadcasted(::typeof($op), a::AbstractZerosVector, b::AbstractZerosVector)
            ax = broadcast_shape(axes(a), axes(b))
            TT = typeof($op(zero(eltype(a)), zero(eltype(b))))
            Zeros(TT, ax)
        end
    end
end

# Need to prevent array-valued fills from broadcasting over entry
_mayberef(x) = Ref(x)
_mayberef(x::Number) = x

function broadcasted(::FillStyle{1}, ::typeof(*), a::AbstractFill, b::AbstractRange)
    broadcast_shape(axes(a), axes(b)) == axes(b) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $b to a Vector first."))
    return broadcasted(*, _mayberef(getindex_value(a)), b)
end

function broadcasted(::FillStyle{1}, ::typeof(*), a::AbstractRange, b::AbstractFill)
    broadcast_shape(axes(a), axes(b)) == axes(a) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $b to a Vector first."))
    return broadcasted(*, a, _mayberef(getindex_value(b)))
end

# support AbstractFill .^ k
broadcasted(op::typeof(Base.literal_pow), ::typeof(^), r::AbstractFill{T,N}, ::Val{k}) where {T,N,k} = broadcasted_fill(op, r, getindex_value(r)^k, axes(r))
broadcasted(op::typeof(Base.literal_pow), ::typeof(^), r::AbstractOnes{T,N}, ::Val{k}) where {T,N,k} = broadcasted_ones(op, r, T, axes(r))
broadcasted(op::typeof(Base.literal_pow), ::typeof(^), r::AbstractZeros{T,N}, ::Val{0}) where {T,N} = broadcasted_ones(op, r, T, axes(r))
broadcasted(op::typeof(Base.literal_pow), ::typeof(^), r::AbstractZeros{T,N}, ::Val{k}) where {T,N,k} = broadcasted_zeros(op, r, T, axes(r))

# supports structured broadcast
if isdefined(LinearAlgebra, :fzero)
    LinearAlgebra.fzero(x::AbstractZeros) = zero(eltype(x))
end
