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
FillStyle{N}(::Val{M}) where {N,M} = FillStyle{M}()
Broadcast.BroadcastStyle(::Type{<:AbstractFill{<:Any,N}}) where {N} = FillStyle{N}()

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
_fillstyle_result(::AbstractFillStyle{M}, ::AbstractFillStyle{N}) where {M,N} = FillStyle{max(M,N)}()

# Obtain the fill value of a broadcasted object by recursively evaluating the fill components
broadcast_getindex_value(f::AbstractFill) = getindex_value(f)
# `transpose`/`adjoint` are recursive, so the wrapper's elements are the parent's fill value with
# the operation applied to it, not the fill value itself. These recurse into the parent rather than
# requiring an `AbstractFill` there, so that they cover every wrapper that `isfill` accepts.
broadcast_getindex_value(f::Transpose) = transpose(broadcast_getindex_value(parent(f)))
broadcast_getindex_value(f::Adjoint) = adjoint(broadcast_getindex_value(parent(f)))
broadcast_getindex_value(x::Number) = x
broadcast_getindex_value(x::Ref) = x[]
function broadcast_getindex_value(bc::Broadcast.Broadcasted)
    bc.f(map(broadcast_getindex_value, bc.args)...)
end

has_static_value(x) = false
has_static_value(x::Union{AbstractZeros, AbstractOnes}) = true
has_static_value(x::Broadcast.Broadcasted) = all(has_static_value, x.args)

# conservative checks for zeros and ones. In most cases, there isn't a zero or unit element to
# compare with
_iszero(x::Union{Number, AbstractArray}) = iszero(x)
_iszero(_) = false
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

# the fill value is computed once and handed to the checks, so that the broadcasted function is
# evaluated exactly once
function _copy_fill(bc)
    v = broadcast_getindex_value(bc)
    if all(has_static_value, bc.args)
        _iszero(v) && return broadcasted_zeros(bc.f, typeof(v), axes(bc), bc.args...)
        _isone(v) && return broadcasted_ones(bc.f, typeof(v), axes(bc), bc.args...)
    end
    return broadcasted_fill(bc.f, v, axes(bc), bc.args...)
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
    # collapsing the fill components may have exposed a rule that returns an array eagerly, in
    # which case there is nothing left to materialize
    bc2 isa Broadcasted || return bc2
    # fallback style
    S = Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{ndims(bc)}}
    copy(convert(S, bc2))
end

function Base.copy(bc::Broadcast.Broadcasted{<:AbstractFillStyle})
    isfill(bc) ? _copy_fill(bc) : _fallback_copy(bc)
end

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
# ours to handle, so ordinary dispatch on the arguments reaches every fill rule, and neither the
# styleless ones nor those attached to a fill style can route back here. `DefaultArrayStyle` is
# deliberately not included: a fill among the arguments never resolves to it here, but a downstream
# style that maps fills onto it would send `broadcasted` straight back and blow the stack.
_dispatch_on_fills(::AbstractFillStyle, ::Val, op, args...) = broadcasted(op, args...)
# A foreign style (e.g. an infinite fill, which `InfiniteArrays` marks as lazy). Re-entering
# style-based dispatch would route straight back here, so apply only the style-independent
# rules and leave the rest to the caller.
function _dispatch_on_fills(::Broadcast.BroadcastStyle, ::Val{N}, op, args...) where {N}
    r = fill_rule(op, args...)
    r === nothing || return r
    # `FillStyle` can't route back here either
    bc = Broadcast.broadcasted(FillStyle{N}(), op, args...)
    bc isa Broadcast.Broadcasted || return bc # a fill-specific method applied
    isfill(bc) && return _copy_fill(bc)
    # a fill-specific method may instead have rewritten the arguments while staying lazy, as the
    # fill-against-a-range rules do. That result already carries the caller's own style, so keep it
    # rather than discarding it for a `DefaultArrayStyle` wrapper around the original arguments.
    bc isa Broadcast.Broadcasted{<:AbstractFillStyle} || return bc
    Broadcast.Broadcasted{DefaultArrayStyle{N}}(op, args)
end

# `Broadcast.broadcast_preserving_zero_d` re-wraps a zero-dimensional result, assuming broadcasting
# unwrapped it to the element. Our rules hand back a container instead, which that re-wrap would
# nest inside a second one, so preserve the shape here rather than restoring it afterwards.
function broadcast_preserving_zero_d(f, As...)
    bc = Base.broadcasted(f, As...)
    # a rule may have applied and returned an array already, in which case the shape is preserved
    bc isa Broadcasted || return bc
    # `materialize` instantiates, which is what checks that the shapes agree
    r = Broadcast.materialize(bc)
    # our own rules keep the container in the zero-dimensional case, but a foreign style handling
    # the broadcast may unwrap it to the element the way Base does
    length(axes(bc)) == 0 && !(r isa AbstractArray) ? Fill(r) : r
end
# Base reaches `broadcast_preserving_zero_d` from `real`/`imag`/`conj`, unary `-`, binary `+`/`-`
# and `*`/`/`/`\` against a number, but it is not private to Base: `Dates` routes
# `Period * AbstractArray` through it too, so extend the function rather than each of its callers.
# `LinearAlgebra` does the same for adjoint and transpose vectors. The third method breaks the tie
# between the first two.
Broadcast.broadcast_preserving_zero_d(f, A::AbstractFill, Bs...) =
    broadcast_preserving_zero_d(f, A, Bs...)
Broadcast.broadcast_preserving_zero_d(f, A, B::AbstractFill, Cs...) =
    broadcast_preserving_zero_d(f, A, B, Cs...)
Broadcast.broadcast_preserving_zero_d(f, A::AbstractFill, B::AbstractFill, Cs...) =
    broadcast_preserving_zero_d(f, A, B, Cs...)
# Base sends a real eltype to `zero`, which names `Zeros` outright, so route it through the
# `broadcasted_zeros` hook instead and let a package customizing that get its own type back.
# `real`/`conj` and every complex-eltype case already reach the hooks: Base returns the argument
# for the former and defers to the broadcast, which applies the hooks, for the latter.
imag(A::AbstractOnes) = broadcasted_zeros(imag, real(eltype(A)), axes(A), A)

### Binary broadcasting

# Default outputs, can overload to customize. The broadcast arguments come last so that a rule may
# pass however many it has, from the one of `exp.(a)` to the whole of a fused `Broadcasted`.
broadcasted_fill(f, val, ax, args...) = Fill(val, ax)
broadcasted_zeros(f, elt, ax, args...) = Zeros{elt}(ax)
broadcasted_ones(f, elt, ax, args...) = Ones{elt}(ax)

function _broadcasted_zeros(f, a, b)
  elt = Base.Broadcast.combine_eltypes(f, (a, b))
  ax = broadcast_shape(axes(a), axes(b))
  return broadcasted_zeros(f, elt, ax, a, b)
end
function _broadcasted_ones(f, a, b)
  elt = Base.Broadcast.combine_eltypes(f, (a, b))
  ax = broadcast_shape(axes(a), axes(b))
  return broadcasted_ones(f, elt, ax, a, b)
end
function _broadcasted_nan(f, a, b)
  val = convert(Base.Broadcast.combine_eltypes(f, (a, b)), NaN)
  ax = broadcast_shape(axes(a), axes(b))
  return broadcasted_fill(f, val, ax, a, b)
end

# In following, need to restrict to <: Number as otherwise we cannot infer zero from type
# TODO: generalise to things like SVector
# `fill_rule` collects the rules that hold whatever the other argument's style is, hence being
# attached to the operation rather than to a style. `nothing` means no rule applies. Holding them on
# a function of their own is what lets the calls a package forwards through `DefaultArrayStyle` reach
# the rules that style-based dispatch would otherwise hand to the caller's own style: `broadcasted`
# cannot be asked whether it has a rule without also running its fallback. Only shapes admitting a
# non-fill argument need an entry: where every argument is a fill, evaluating the operation on the
# fill values suffices.
fill_rule(op, args...) = nothing
for T in (:Number, :(AbstractArray{<:Number}), :(Base.Broadcast.Broadcasted))
    for (op, a, b) in ((:*, :AbstractZeros, T), (:/, :AbstractZeros, T),
                       (:*, T, :AbstractZeros), (:\, T, :AbstractZeros))
        @eval begin
            fill_rule(::typeof($op), a::$a, b::$b) = _broadcasted_zeros($op, a, b)
            broadcasted(::typeof($op), a::$a, b::$b) = fill_rule($op, a, b)
        end
    end
end
for (op, zeros_rule) in ((:*, :_broadcasted_zeros), (:/, :_broadcasted_nan), (:\, :_broadcasted_nan))
    @eval begin
        # two fills, so the rule is only needed to disambiguate the one-sided entries above
        fill_rule(::typeof($op), a::AbstractZeros, b::AbstractZeros) = $zeros_rule($op, a, b)
        broadcasted(::typeof($op), a::AbstractZeros, b::AbstractZeros) = fill_rule($op, a, b)
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

# Unlike the rules above, these return the other argument itself where they can, so they must not
# pre-empt a style that would have built a container of its own. They are attached to the fill
# styles, which win only once every other style has deferred to `DefaultArrayStyle`, and `fill_rule`
# makes them reachable for the calls a package forwards through `DefaultArrayStyle`.
for op in (:+, :-)
    @eval begin
        function fill_rule(::typeof($op), a::AbstractVector, b::AbstractZerosVector)
            ax = broadcast_shape(axes(a), axes(b))
            # the size, rather than `a` itself, which the rule may be handed unrendered
            ax == axes(a) || throw(ArgumentError(LazyString("cannot broadcast an array with size ",
                size(a), " with ", b, ". Convert ", b, " to a Vector first.")))
            TT = typeof($op(zero(eltype(a)), zero(eltype(b))))
            # Use `TT ∘ (+)` to fix AD issues with `broadcasted(TT, x)`
            eltype(a) === TT ? a : broadcasted(TT ∘ (+), a)
        end
        function fill_rule(::typeof($op), a::AbstractZerosVector, b::AbstractVector)
            ax = broadcast_shape(axes(a), axes(b))
            ax == axes(b) || throw(ArgumentError(LazyString("cannot broadcast ", a,
                " with an array with size ", size(b), ". Convert ", a, " to a Vector first.")))
            TT = typeof($op(zero(eltype(a)), zero(eltype(b))))
            $op === (+) && eltype(b) === TT ? b : broadcasted(TT ∘ ($op), b)
        end
        function fill_rule(::typeof($op), a::AbstractZerosVector, b::AbstractZerosVector)
            ax = broadcast_shape(axes(a), axes(b))
            TT = typeof($op(zero(eltype(a)), zero(eltype(b))))
            broadcasted_zeros($op, TT, ax, a, b)
        end
        broadcasted(::AbstractFillStyle{1}, ::typeof($op), a::AbstractVector, b::AbstractZerosVector) =
            fill_rule($op, a, b)
        broadcasted(::AbstractFillStyle{1}, ::typeof($op), a::AbstractZerosVector, b::AbstractVector) =
            fill_rule($op, a, b)
        broadcasted(::AbstractFillStyle{1}, ::typeof($op), a::AbstractZerosVector, b::AbstractZerosVector) =
            fill_rule($op, a, b)
    end
end

# Need to prevent array-valued fills from broadcasting over entry
_mayberef(x) = Ref(x)
_mayberef(x::Number) = x

# Scaling a range by a constant leaves a range, and so does shifting it, so rewrite both the same
# way and let the range decide. `Zeros` needs nothing of its own here, being a fill whose value is
# zero, but a range is a vector and `Zeros` is a fill, so these overlap the `Zeros` rules without
# either being the more specific. The last two only break that tie, and do the same thing.
function _rewrite_range(op, a::AbstractFill, b::AbstractRange)
    broadcast_shape(axes(a), axes(b)) == axes(b) || throw(ArgumentError(LazyString("Cannot broadcast ", a, " and ", b, ". Convert ", b, " to a Vector first.")))
    return broadcasted(op, _mayberef(getindex_value(a)), b)
end
function _rewrite_range(op, a::AbstractRange, b::AbstractFill)
    broadcast_shape(axes(a), axes(b)) == axes(a) || throw(ArgumentError(LazyString("Cannot broadcast ", a, " and ", b, ". Convert ", b, " to a Vector first.")))
    return broadcasted(op, a, _mayberef(getindex_value(b)))
end
for op in (:*, :+, :-)
    @eval begin
        broadcasted(::FillStyle{1}, ::typeof($op), a::AbstractFill, b::AbstractRange) = _rewrite_range($op, a, b)
        broadcasted(::FillStyle{1}, ::typeof($op), a::AbstractRange, b::AbstractFill) = _rewrite_range($op, a, b)
        broadcasted(::FillStyle{1}, ::typeof($op), a::AbstractZerosVector, b::AbstractRange) = _rewrite_range($op, a, b)
        broadcasted(::FillStyle{1}, ::typeof($op), a::AbstractRange, b::AbstractZerosVector) = _rewrite_range($op, a, b)
    end
end

# support AbstractFill .^ k
broadcasted(op::typeof(Base.literal_pow), ::typeof(^), r::AbstractFill{T,N}, ::Val{k}) where {T,N,k} = broadcasted_fill(op, getindex_value(r)^k, axes(r), r)
broadcasted(op::typeof(Base.literal_pow), ::typeof(^), r::AbstractOnes{T,N}, ::Val{k}) where {T,N,k} = broadcasted_ones(op, T, axes(r), r)
broadcasted(op::typeof(Base.literal_pow), ::typeof(^), r::AbstractZeros{T,N}, ::Val{0}) where {T,N} = broadcasted_ones(op, T, axes(r), r)
broadcasted(op::typeof(Base.literal_pow), ::typeof(^), r::AbstractZeros{T,N}, ::Val{k}) where {T,N,k} = broadcasted_zeros(op, T, axes(r), r)
fill_rule(op::typeof(Base.literal_pow), x::typeof(^), r::AbstractFill, y::Val) = broadcasted(op, x, r, y)

# supports structured broadcast
if isdefined(LinearAlgebra, :fzero)
    LinearAlgebra.fzero(x::AbstractZeros) = zero(eltype(x))
end
