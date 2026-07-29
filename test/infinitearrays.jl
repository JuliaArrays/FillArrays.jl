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

    # Broadcasting, following what the `InfiniteArrays` package does through the `LazyArrayStyle`
    # of `LazyArrays`: an infinite array can't be materialized, so it uses a style of its own that
    # wins against `DefaultArrayStyle`, and fill arguments are forwarded to `DefaultArrayStyle`
    # explicitly so that they keep being simplified by `FillArrays`.
    struct InfiniteArrayStyle{N} <: AbstractArrayStyle{N} end
    InfiniteArrayStyle{M}(::Val{N}) where {M,N} = InfiniteArrayStyle{N}()

    BroadcastStyle(::Type{<:AbstractInfUnitRange}) = InfiniteArrayStyle{1}()
    for typ in (:Ones, :Zeros, :Fill)
        @eval BroadcastStyle(::Type{<:$typ{T,N,<:Tuple{OneToInf,Vararg{OneToInf}}}}) where {T,N} =
            InfiniteArrayStyle{N}()
    end

    # an infinite array can't be materialized, so the lazy `Broadcasted` stands in for the
    # `BroadcastArray` that `LazyArrays` would return here
    Base.copy(bc::Broadcasted{<:InfiniteArrayStyle}) = bc

    # the forwards that `LazyArrays` defines, which cover the one- and two-argument forms
    broadcasted(::InfiniteArrayStyle{N}, op, r::AbstractFill{T,N}) where {T,N} =
        broadcast(DefaultArrayStyle{N}(), op, r)
    broadcasted(::InfiniteArrayStyle{N}, op, r::AbstractFill{T,N}, x::Number) where {T,N} =
        broadcast(DefaultArrayStyle{N}(), op, r, x)
    broadcasted(::InfiniteArrayStyle{N}, op, x::Number, r::AbstractFill{T,N}) where {T,N} =
        broadcast(DefaultArrayStyle{N}(), op, x, r)
    broadcasted(::InfiniteArrayStyle{N}, op, r1::AbstractFill{T,N}, r2::AbstractFill{V,N}) where {T,V,N} =
        broadcast(DefaultArrayStyle{N}(), op, r1, r2)
    # `x .^ k` lowers to a three-argument `literal_pow` broadcast, which the forwards above don't
    # match, so it is forwarded explicitly as well
    broadcasted(::InfiniteArrayStyle{N}, op::typeof(Base.literal_pow), x::Base.RefValue{typeof(^)},
            r::AbstractFill{T,N}, y::Base.RefValue{<:Val}) where {T,N} =
        broadcast(DefaultArrayStyle{N}(), op, x, r, y)
    # Ranges are only forwarded for the fills that are known to simplify against them, as this
    # helper has no lazy infinite range to hold a result such as `Fill(2, ∞) .* (1:∞)`.
    # `LazyArrays` forwards these for an `AbstractFill`, as `InfiniteArrays` provides such a range.
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
