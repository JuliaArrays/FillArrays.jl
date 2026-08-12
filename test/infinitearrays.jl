# Infinite Arrays implementation from
# https://github.com/JuliaLang/julia/blob/master/test/testhelpers/InfiniteArrays.jl
module InfiniteArrays
    using Infinities
    using FillArrays
    using FillArrays: AbstractFill, AbstractOnes, AbstractZeros
    using Base.Broadcast: AbstractArrayStyle, Broadcasted, DefaultArrayStyle
    import Base.Broadcast: BroadcastStyle, broadcasted
    export OneToInf

    abstract type AbstractInfUnitRange{T<:Real} <: AbstractUnitRange{T} end
    Base.length(r::AbstractInfUnitRange) = ℵ₀
    Base.size(r::AbstractInfUnitRange) = (ℵ₀,)
    Base.last(r::AbstractInfUnitRange) = ℵ₀
    Base.axes(r::AbstractInfUnitRange) = (OneToInf(),)

    Base.IteratorSize(::Type{<:AbstractInfUnitRange}) = Base.IsInfinite()

    """
        OneToInf(n)
    Define an `AbstractInfUnitRange` that behaves like `1:∞`, with the added
    distinction that the limits are guaranteed (by the type system) to
    be 1 and ∞.
    """
    struct OneToInf{T<:Integer} <: AbstractInfUnitRange{T} end

    OneToInf() = OneToInf{Int}()

    Base.axes(r::OneToInf) = (r,)
    Base.first(r::OneToInf{T}) where {T} = oneunit(T)
    Base.oneto(::InfiniteCardinal{0}) = OneToInf()

    struct InfUnitRange{T<:Real} <: AbstractInfUnitRange{T}
        start::T
    end
    Base.first(r::InfUnitRange) = r.start
    InfUnitRange(a::InfUnitRange) = a
    InfUnitRange{T}(a::AbstractInfUnitRange) where T<:Real = InfUnitRange{T}(first(a))
    InfUnitRange(a::AbstractInfUnitRange{T}) where T<:Real = InfUnitRange{T}(first(a))
    Base.:(:)(start::T, stop::InfiniteCardinal{0}) where {T<:Integer} = InfUnitRange{T}(start)
    function getindex(v::InfUnitRange{T}, i::Integer) where T
        @boundscheck i > 0 || Base.throw_boundserror(v, i)
        convert(T, first(v) + i - 1)
    end

    # neither a fill nor a range, so a rule absorbing it can't be recovered from fill values
    struct InfVector <: AbstractVector{Int} end
    Base.axes(::InfVector) = (OneToInf(),)
    Base.size(::InfVector) = (ℵ₀,)
    Base.getindex(::InfVector, i::Integer) = 2i
    # it holds no storage to be written through, so adding `Zeros` may hand it straight back
    FillArrays.has_mutable_storage(::InfVector) = false

    # Broadcasting as `InfiniteArrays` does it through the `LazyArrayStyle` of `LazyArrays`: a
    # style of its own that wins against `DefaultArrayStyle`, with fills forwarded back to
    # `DefaultArrayStyle` so that `FillArrays` keeps simplifying them.
    struct InfiniteArrayStyle{N} <: AbstractArrayStyle{N} end
    InfiniteArrayStyle{M}(::Val{N}) where {M,N} = InfiniteArrayStyle{N}()

    BroadcastStyle(::Type{<:AbstractInfUnitRange}) = InfiniteArrayStyle{1}()
    BroadcastStyle(::Type{InfVector}) = InfiniteArrayStyle{1}()
    for typ in (:Ones, :Zeros, :Fill)
        @eval BroadcastStyle(::Type{<:$typ{T,N,<:Tuple{OneToInf,Vararg{OneToInf}}}}) where {T,N} =
            InfiniteArrayStyle{N}()
    end

    # stands in for the `BroadcastArray` that `LazyArrays` would return
    Base.copy(bc::Broadcasted{<:InfiniteArrayStyle}) = bc

    # the one- and two-argument forwards that `LazyArrays` defines
    broadcasted(::InfiniteArrayStyle{N}, op, r::AbstractFill{T,N}) where {T,N} =
        broadcast(DefaultArrayStyle{N}(), op, r)
    broadcasted(::InfiniteArrayStyle{N}, op, r::AbstractFill{T,N}, x::Number) where {T,N} =
        broadcast(DefaultArrayStyle{N}(), op, r, x)
    broadcasted(::InfiniteArrayStyle{N}, op, x::Number, r::AbstractFill{T,N}) where {T,N} =
        broadcast(DefaultArrayStyle{N}(), op, x, r)
    broadcasted(::InfiniteArrayStyle{N}, op, r1::AbstractFill{T,N}, r2::AbstractFill{V,N}) where {T,V,N} =
        broadcast(DefaultArrayStyle{N}(), op, r1, r2)
    # `x .^ k` lowers to a three-argument `literal_pow` that the forwards above don't match
    broadcasted(::InfiniteArrayStyle{N}, op::typeof(Base.literal_pow), x::Base.RefValue{typeof(^)},
            r::AbstractFill{T,N}, y::Base.RefValue{<:Val}) where {T,N} =
        broadcast(DefaultArrayStyle{N}(), op, x, r, y)
    # Only the fills that simplify against a range are forwarded: `LazyArrays` forwards any
    # `AbstractFill`, but this helper has no lazy range to hold `Fill(2, ∞) .* (1:∞)`.
    broadcasted(::InfiniteArrayStyle{1}, ::typeof(*), a::AbstractOnes{<:Any,1}, b::AbstractRange) =
        broadcast(DefaultArrayStyle{1}(), *, a, b)
    broadcasted(::InfiniteArrayStyle{1}, ::typeof(*), a::AbstractRange, b::AbstractOnes{<:Any,1}) =
        broadcast(DefaultArrayStyle{1}(), *, a, b)
    for op in (:*, :+, :-)
        @eval begin
            broadcasted(::InfiniteArrayStyle{1}, ::typeof($op), a::AbstractZeros{<:Any,1}, b::AbstractRange) =
                broadcast(DefaultArrayStyle{1}(), $op, a, b)
            broadcasted(::InfiniteArrayStyle{1}, ::typeof($op), a::AbstractRange, b::AbstractZeros{<:Any,1}) =
                broadcast(DefaultArrayStyle{1}(), $op, a, b)
        end
    end
end
