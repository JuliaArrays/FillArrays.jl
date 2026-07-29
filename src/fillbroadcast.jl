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
            LazyString("dimensions must match: a has dims ", r1, ", b has dims ", axes(r))))
        end
        return false
    end
end

# this preserves the init cases of Base.mapreduce
_fill_reduce_empty_iter(f, op, A) = Base.reduce_empty_iter(Base._xfadjoint(Base.BottomRF(op), Base.Generator(f, A))...)
# unlike a standard array, we have a well-defined notion of max/min for FillArrays
_fill_reduce_empty_iter(f, ::Union{typeof(max), typeof(min)}, A) = f(getindex_value(A))


# these special cases can be computed exactly. 
# We special-case zeros value to avoid ∞ * 0 in InfiniteArrays.jl
_foldl_length_op(f, ::Union{typeof(max), typeof(min), typeof(&), typeof(|)}, A) = f(getindex_value(A))

function _foldl_length_op(f, op::Union{typeof(+), typeof(add_sum)}, A)
    v = f(getindex_value(A))
    iszero(v) && return op(v, v) # get type right
    length(A)*v
end

function _foldl_length_op(f, op::Union{typeof(*), typeof(mul_prod)}, A)
    v = f(getindex_value(A))
    (iszero(v) || isone(v)) && return op(v, v) # get type right
    v^length(A)
end

### mapreduce
# Fast special cases
for mapfold in (:(Base.mapfoldl_impl), :(Base.mapfoldr_impl)), op in (:max, :min, :&, :|, :+, :add_sum, :*, :mul_prod)
    @eval function $mapfold(f, ::typeof($op), nt, A::AbstractFill)
        if nt isa Base._InitialValue
            isempty(A) && return _fill_reduce_empty_iter(f, $op, A)
            _foldl_length_op(f, $op, A) # multiplication promotes type a la +, add_sum
        else
            isempty(A) && return nt
            $op(nt, _foldl_length_op(f, $op, A))
        end
    end
end

function Base.mapfoldl_impl(f, op, nt, A::AbstractFill)
    fval = f(getindex_value(A))
    out = if nt isa Base._InitialValue
        isempty(A) && return _fill_reduce_empty_iter(f, op, A)
        fval
    elseif isempty(A)
        nt
    else
        op(nt, fval)
    end
    for _ in 2:length(A)
        out = op(out, fval)
    end
    out
end

function Base.mapfoldr_impl(f, op, nt, A::AbstractFill)
    fval = f(getindex_value(A))
    out = if nt isa Base._InitialValue
        isempty(A) && return _fill_reduce_empty_iter(f, op, A)
        fval
    elseif isempty(A)
        nt
    else
        op(nt, fval)
    end
    for _ in length(A)-1:-1:1
        out = op(fval, out)
    end
    out
end


mapreduce(f, op, A::AbstractFill; dims=:, init=Base._InitialValue()) = _fill_mapreduce_dim(f, op, init, A, dims)
_fill_mapreduce_dim(f, op, init, A, ::Colon) = mapfoldl(f, op, A; init=init) # foldl is fast

# Opposite of Base.reduced_indices
# Based on base/reducedim.jl
# for reductions that expand ≠0 dims to 1
function _check_valid_region(region)
    for d in region
        isa(d, Integer) || throw(ArgumentError("reduced dimension(s) must be integers"))
        Int(d) < 1 && throw(ArgumentError("region dimension(s) must be ≥ 1, got $d"))
    end
end

unreduced_indices(a, region) = unreduced_indices(axes(a), region)
function unreduced_indices(axs::Base.Indices{N}, region) where N
    _check_valid_region(region)
    ntuple(d -> !(d in region) ? Base.reduced_index(axs[d]) : axs[d], Val(N))
end

function _fill_mapreduce_dim(f, op, init, A, dims)
    A_red = A[unreduced_indices(A, dims)...] # reduce A, still a Fill
    Fill(mapreduce(f, op, A_red; init=init), ntuple(d -> d in dims ? Base.OneTo(1) : axes(A,d), ndims(A)))
end

# map for AbstractFills returns an AbstractFill so just call reduce
mapreduce(f, op, A::AbstractFill, B::AbstractFill, Cs::AbstractFill...; kw...) = reduce(op, map(f, A, B, Cs...); kw...)

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

# A fill style resolves conflicts the way `DefaultArrayStyle` of the same dimension does, so
# that styles winning against or deferring to the default one need no rule of their own. Only
# these two `where`-shapes are defined, leaving the rules below unambiguous with them.
Broadcast.BroadcastStyle(a::Broadcast.AbstractArrayStyle{Any}, b::AbstractFillStyle) = _fillstyle_result(a, b)
Broadcast.BroadcastStyle(a::Broadcast.AbstractArrayStyle{M}, b::AbstractFillStyle{N}) where {M,N} = _fillstyle_result(a, b)

_fillstyle_result(a::Broadcast.AbstractArrayStyle, b::AbstractFillStyle{N}) where {N} =
    _fillstyle_defer(Broadcast.result_style(a, DefaultArrayStyle{N}()), b)
# the other style defers, so the fill structure may be preserved
_fillstyle_defer(::DefaultArrayStyle{M}, b::AbstractFillStyle) where {M} = typeof(b)(Val(M))
_fillstyle_defer(a::Broadcast.BroadcastStyle, ::AbstractFillStyle) = a
# `FillStyle` wins within the family, being the less specific of the two
_fillstyle_result(::AbstractFillStyle{M}, ::AbstractFillStyle{N}) where {M,N} = FillStyle{max(M,N)}()
_fillstyle_result(::ZerosStyle{M}, ::ZerosStyle{N}) where {M,N} = ZerosStyle{max(M,N)}()

# Obtain the fill value of a broadcasted object by recursively evaluating the fill components
broadcast_getindex_value(f::AbstractFill) = getindex_value(f)
broadcast_getindex_value(f::Transpose{<:Any,<:AbstractFill}) = getindex_value(parent(f))
broadcast_getindex_value(f::Adjoint{<:Any,<:AbstractFill}) = getindex_value(parent(f))
broadcast_getindex_value(x::Number) = x
broadcast_getindex_value(x::Ref) = x[]
function broadcast_getindex_value(bc::Broadcast.Broadcasted)
    bc.f(map(broadcast_getindex_value, bc.args)...)
end

has_static_value(x) = false
has_static_value(x::Union{AbstractZeros, AbstractOnes}) = true
has_static_value(x::Broadcast.Broadcasted) = all(has_static_value, x.args)

# _iszeros and _isones are conservative checks for zeros and ones,
# which are used to determine if a broadcasted object is a Fill, Zeros or Ones.
function _iszeros(bc::Broadcast.Broadcasted)
    all(has_static_value, bc.args) && _iszero(broadcast_getindex_value(bc))
end
# conservative check for zeros. In most cases, there isn't a zero element to compare with
_iszero(x::Union{Number, AbstractArray}) = iszero(x)
_iszero(_) = false

function _isones(bc::Broadcast.Broadcasted)
    all(has_static_value, bc.args) && _isone(broadcast_getindex_value(bc))
end
# conservative check for ones. In most cases, there isn't a unit element to compare with
_isone(x::Union{Number, AbstractArray}) = isone(x)
_isone(_) = false

# wrappers that are equivalent to an `AbstractFill` may opt in to the broadcasting behavior
# of `AbstractFill` by specializing `isfill` and `broadcast_getindex_value`
isfill(bc::Broadcast.Broadcasted) = all(isfill, bc.args)
isfill(f::AbstractFill) = true
isfill(f::Transpose) = isfill(parent(f))
isfill(f::Adjoint) = isfill(parent(f))
isfill(f::Number) = true
isfill(f::Ref) = true
isfill(::Any) = false

function _copy_fill(bc)
    v = broadcast_getindex_value(bc)
    if _iszeros(bc)
        return Zeros(typeof(v), axes(bc))
    elseif _isones(bc)
        return Ones(typeof(v), axes(bc))
    end
    return Fill(v, axes(bc))
end

# recursively copy the purely fill components
function _preprocess_fill(bc::Broadcast.Broadcasted{<:AbstractFillStyle})
    isfill(bc) ? _copy_fill(bc) : Broadcast.broadcasted(bc.f, map(_preprocess_fill, bc.args)...)
end
_preprocess_fill(bc::Broadcast.Broadcasted) = Broadcast.broadcasted(bc.f, map(_preprocess_fill, bc.args)...)
_preprocess_fill(x) = x

function _fallback_copy(bc)
    # copy the purely fill components
    bc2 = Base.broadcasted(bc.f, map(_preprocess_fill, bc.args)...)
    # fallback style
    S = Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{ndims(bc)}}
    copy(convert(S, bc2))
end

function Base.copy(bc::Broadcast.Broadcasted{<:AbstractFillStyle})
    isfill(bc) ? _copy_fill(bc) : _fallback_copy(bc)
end
# make the zero-dimensional case consistent with Base
Base.copy(bc::Broadcast.Broadcasted{<:AbstractFillStyle{0}}) = _fallback_copy(bc)

# Packages with a style of their own (e.g. LazyArrays) opt out of it for fills by forwarding to
# `DefaultArrayStyle`. The fill rules are no longer attached to that style, so re-dispatch such
# calls on the arguments alone and keep returning a fill.
broadcasted(::DefaultArrayStyle{N}, op, r::AbstractFill) where {N} = _dispatch_on_fills(Val(N), op, r)
broadcasted(::DefaultArrayStyle{N}, op, a::AbstractFill, b) where {N} = _dispatch_on_fills(Val(N), op, a, b)
broadcasted(::DefaultArrayStyle{N}, op, a, b::AbstractFill) where {N} = _dispatch_on_fills(Val(N), op, a, b)
broadcasted(::DefaultArrayStyle{N}, op, a::AbstractFill, b::AbstractFill) where {N} = _dispatch_on_fills(Val(N), op, a, b)
# `x .^ k` lowers to a three-argument `literal_pow` broadcast, matching none of the shapes above.
# Unwrapping the `Ref`s lets the styleless rules apply, so the style comes from the fill alone;
# the `Ref`s are zero-dimensional and wouldn't have contributed to it anyway.
broadcasted(::DefaultArrayStyle{N}, op::typeof(Base.literal_pow), x::Base.RefValue{typeof(^)},
        r::AbstractFill, y::Base.RefValue{<:Val}) where {N} =
    _dispatch_on_fills(Broadcast.combine_styles(r), Val(N), op, x[], r, y[])

_dispatch_on_fills(v::Val, op, args...) = _dispatch_on_fills(Broadcast.combine_styles(args...), v, op, args...)
# ours to handle, and the style can't route the styleless methods back here
_dispatch_on_fills(::Union{AbstractFillStyle,DefaultArrayStyle}, ::Val, op, args...) = broadcasted(op, args...)
# A foreign style (e.g. an infinite fill, which `InfiniteArrays` marks as lazy). Re-entering
# style-based dispatch would route straight back here, so apply only the style-independent
# rules and leave the rest to the caller.
function _dispatch_on_fills(::Broadcast.BroadcastStyle, ::Val{N}, op, args...) where {N}
    has_fill_rule(op, args...) && return broadcasted(op, args...)
    # `FillStyle` can't route back here either
    bc = Broadcast.broadcasted(FillStyle{N}(), op, args...)
    bc isa Broadcast.Broadcasted || return bc # a fill-specific method applied
    isfill(bc) ? _copy_fill(bc) : Broadcast.Broadcasted{DefaultArrayStyle{N}}(op, args)
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
# These rules hold whatever the other argument's style is, hence being attached to the operation
# rather than to a style. `has_fill_rule` records that one applies, and routes the calls that
# packages forward through `DefaultArrayStyle`. Only shapes admitting a non-fill argument need an
# entry: where every argument is a fill, evaluating the operation on the fill values suffices.
has_fill_rule(op, args...) = false
for T in (:(AbstractFill{<:Number}), :Number, :AbstractOnes, :AbstractRange, :(AbstractArray{<:Number}), :(Base.Broadcast.Broadcasted))
    for op in (:*, :/)
        @eval begin
            broadcasted(::typeof($op), a::AbstractZeros, b::$T) = _broadcasted_zeros($op, a, b)
            has_fill_rule(::typeof($op), ::AbstractZeros, ::$T) = true
        end
    end
    for op in (:*, :\)
        @eval begin
            broadcasted(::typeof($op), a::$T, b::AbstractZeros) = _broadcasted_zeros($op, a, b)
            has_fill_rule(::typeof($op), ::$T, ::AbstractZeros) = true
        end
    end
end
broadcasted(::typeof(*), a::AbstractZeros, b::AbstractZeros) = _broadcasted_zeros(*, a, b)
broadcasted(::typeof(/), a::AbstractZeros, b::AbstractZeros) = _broadcasted_nan(/, a, b)
broadcasted(::typeof(\), a::AbstractZeros, b::AbstractZeros) = _broadcasted_nan(\, a, b)
for op in (:*, :/, :\)
    @eval begin
        # two fills, so only needed to disambiguate the one-sided entries above
        has_fill_rule(::typeof($op), ::AbstractZeros, ::AbstractZeros) = true
        broadcasted(::typeof($op), a::AbstractOnes, b::AbstractOnes) = _broadcasted_ones($op, a, b)
    end
end

# special case due to missing converts for ranges
_range_convert(::Type{AbstractVector{T}}, a::AbstractRange{T}) where T = a
# without this the `AbstractUnitRange` method below wins over the `AbstractRange{T}` one above,
# and the endpoints it converts may not be representable, as for an infinite range
_range_convert(::Type{AbstractVector{T}}, a::AbstractUnitRange{T}) where T = a
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
    broadcast_shape(axes(a), axes(b)) == axes(b) || throw(ArgumentError(LazyString("Cannot broadcast ", a, " and ", b, ". Convert ", b, " to a Vector first.")))
    TT = typeof(zero(eltype(a)) * zero(eltype(b)))
    return _range_convert(AbstractVector{TT}, b)
end

function broadcasted(::FillStyle{1}, ::typeof(*), a::AbstractRange, b::AbstractOnes)
    broadcast_shape(axes(a), axes(b)) == axes(a) || throw(ArgumentError(LazyString("Cannot broadcast ", a, " and ", b, ". Convert ", b, " to a Vector first.")))
    TT = typeof(zero(eltype(a)) * zero(eltype(b)))
    return _range_convert(AbstractVector{TT}, a)
end

for op in (:+, :-)
    @eval begin
        function broadcasted(::typeof($op), a::AbstractVector, b::AbstractZerosVector)
            ax = broadcast_shape(axes(a), axes(b))
            ax == axes(a) || throw(ArgumentError(LazyString("cannot broadcast an array with size ", size(a), " with ", b)))
            TT = typeof($op(zero(eltype(a)), zero(eltype(b))))
            # Use `TT ∘ (+)` to fix AD issues with `broadcasted(TT, x)`
            eltype(a) === TT ? a : broadcasted(TT ∘ (+), a)
        end
        function broadcasted(::typeof($op), a::AbstractZerosVector, b::AbstractVector)
            ax = broadcast_shape(axes(a), axes(b))
            ax == axes(b) || throw(ArgumentError(LazyString("cannot broadcast ", a, " with an array with size ", size(b))))
            TT = typeof($op(zero(eltype(a)), zero(eltype(b))))
            $op === (+) && eltype(b) === TT ? b : broadcasted(TT ∘ ($op), b)
        end
        function broadcasted(::typeof($op), a::AbstractZerosVector, b::AbstractZerosVector)
            ax = broadcast_shape(axes(a), axes(b))
            TT = typeof($op(zero(eltype(a)), zero(eltype(b))))
            Zeros(TT, ax)
        end
        has_fill_rule(::typeof($op), ::AbstractVector, ::AbstractZerosVector) = true
        has_fill_rule(::typeof($op), ::AbstractZerosVector, ::AbstractVector) = true
        # as above, only to disambiguate
        has_fill_rule(::typeof($op), ::AbstractZerosVector, ::AbstractZerosVector) = true
    end
end

# Need to prevent array-valued fills from broadcasting over entry
_mayberef(x) = Ref(x)
_mayberef(x::Number) = x

function broadcasted(::FillStyle{1}, ::typeof(*), a::AbstractFill, b::AbstractRange)
    broadcast_shape(axes(a), axes(b)) == axes(b) || throw(ArgumentError(LazyString("Cannot broadcast ", a, " and ", b, ". Convert ", b, " to a Vector first.")))
    return broadcasted(*, _mayberef(getindex_value(a)), b)
end

function broadcasted(::FillStyle{1}, ::typeof(*), a::AbstractRange, b::AbstractFill)
    broadcast_shape(axes(a), axes(b)) == axes(a) || throw(ArgumentError(LazyString("Cannot broadcast ", a, " and ", b, ". Convert ", b, " to a Vector first.")))
    return broadcasted(*, a, _mayberef(getindex_value(b)))
end

# support AbstractFill .^ k
broadcasted(op::typeof(Base.literal_pow), ::typeof(^), r::AbstractFill{T,N}, ::Val{k}) where {T,N,k} = broadcasted_fill(op, r, getindex_value(r)^k, axes(r))
broadcasted(op::typeof(Base.literal_pow), ::typeof(^), r::AbstractOnes{T,N}, ::Val{k}) where {T,N,k} = broadcasted_ones(op, r, T, axes(r))
broadcasted(op::typeof(Base.literal_pow), ::typeof(^), r::AbstractZeros{T,N}, ::Val{0}) where {T,N} = broadcasted_ones(op, r, T, axes(r))
broadcasted(op::typeof(Base.literal_pow), ::typeof(^), r::AbstractZeros{T,N}, ::Val{k}) where {T,N,k} = broadcasted_zeros(op, r, T, axes(r))
has_fill_rule(::typeof(Base.literal_pow), ::typeof(^), ::AbstractFill, ::Val) = true

# supports structured broadcast
if isdefined(LinearAlgebra, :fzero)
    LinearAlgebra.fzero(x::AbstractZeros) = zero(eltype(x))
end
