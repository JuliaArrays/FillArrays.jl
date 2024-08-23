using FillArrays, LinearAlgebra, PDMats, SparseArrays, StaticArrays, ReverseDiff, Random, Test, Statistics, Quaternions

import FillArrays: AbstractFill, RectDiagonal, SquareEye

using Documenter
DocMeta.setdocmeta!(FillArrays, :DocTestSetup, :(using FillArrays))
doctest(FillArrays; manual = false)

include("aqua.jl")

include("infinitearrays.jl")
import .InfiniteArrays

# we may use this instead of rand(n) to generate deterministic arrays
oneton(T::Type, sz...) = reshape(T.(1:prod(sz)), sz)
oneton(sz...) = oneton(Float64, sz...)

stringmime(args...) = sprint(show, args...)

@testset "fill array constructors and convert" begin
    for (Typ, funcs) in ((Zeros, zeros), (Ones, ones))
        @test Typ((-1,5)) == Typ((0,5))
        @test Typ(5) isa AbstractVector{Float64}
        @test Typ(5,5) isa AbstractMatrix{Float64}
        @test Typ(5) == Typ((5,))
        @test Typ(5,5) == Typ((5,5))
        @test eltype(Typ(5,5)) == Float64

        for T in (Int, Float64)
            Z = Typ{T}(5)
            @test Typ(T, 5) ≡ Z
            @test eltype(Z) == T
            @test Array(Z) == funcs(T,5)
            @test Array{T}(Z) == funcs(T,5)
            @test Array{T,1}(Z) == funcs(T,5)

            @test convert(AbstractArray,Z) ≡ Z
            @test convert(AbstractArray{T},Z) ≡ AbstractArray{T}(Z) ≡ Z
            @test convert(AbstractVector{T},Z) ≡ AbstractVector{T}(Z) ≡ Z
            @test convert(AbstractFill{T},Z) ≡ AbstractFill{T}(Z) ≡ Z

            @test Typ{T,1}(2ones(T,5)) == Z
            @test Typ{T}(2ones(T,5)) == Z
            @test Typ(2ones(T,5)) == Z

            Z = Typ{T}(5, 5)
            @test Typ(T, 5, 5) ≡ Z
            @test eltype(Z) == T
            @test Array(Z) == funcs(T,5,5)
            @test Array{T}(Z) == funcs(T,5,5)
            @test Array{T,2}(Z) == funcs(T,5,5)

            @test convert(AbstractArray,Z) ≡ convert(AbstractFill,Z) ≡ Z
            @test convert(AbstractArray{T},Z) ≡ convert(AbstractFill{T},Z) ≡ AbstractArray{T}(Z) ≡ Z
            @test convert(AbstractMatrix{T},Z) ≡ convert(AbstractFill{T,2},Z) ≡ AbstractMatrix{T}(Z) ≡ Z

            @test_throws Exception convert(Fill{Float64}, [1,1,2])
            @test_throws Exception convert(Fill, [])
            @test convert(Fill{Float64}, [1,1,1]) ≡ Fill(1.0, 3)
            @test convert(Fill, Float64[1,1,1]) ≡ Fill(1.0, 3)
            @test convert(Fill{Float64}, Fill(1.0,2)) ≡ Fill(1.0, 2) # was ambiguous
            @test convert(Fill{Int}, Ones(20)) ≡ Fill(1, 20)
            @test convert(Fill{Int,1}, Ones(20)) ≡ Fill(1, 20)
            @test convert(Fill{Int,1,Tuple{Base.OneTo{Int}}}, Ones(20)) ≡ Fill(1, 20)
            @test convert(AbstractFill{Int}, Ones(20)) ≡ AbstractFill{Int}(Ones(20)) ≡ Ones{Int}(20)
            @test convert(AbstractFill{Int,1}, Ones(20)) ≡ AbstractFill{Int,1}(Ones(20)) ≡ Ones{Int}(20)
            @test convert(AbstractFill{Int,1,Tuple{Base.OneTo{Int}}}, Ones(20)) ≡ AbstractFill{Int,1,Tuple{Base.OneTo{Int}}}(Ones(20)) ≡ Ones{Int}(20)

            @test Typ{T,2}(2ones(T,5,5)) ≡ Typ{T}(5,5)
            @test Typ{T}(2ones(T,5,5)) ≡ Typ{T}(5,5)
            @test Typ(2ones(T,5,5)) ≡ Typ{T}(5,5)

            z = Typ{T}()[]
            @test convert(Typ, Fill(z,2)) === Typ{T}(2)
            z = Typ{Int8}()[]
            @test convert(Typ{T}, Fill(z,2)) === Typ{T}(2)
            @test convert(Typ{T,1}, Fill(z,2)) === Typ{T}(2)
            @test convert(Typ{T,1,Tuple{Base.OneTo{Int}}}, Fill(z,2)) === Typ{T}(2)
            @test_throws ArgumentError convert(Typ, Fill(2,2))

            @test Typ(Z) ≡ Typ{T}(Z) ≡ Typ{T,2}(Z) ≡ typeof(Z)(Z) ≡ Z

            @test AbstractArray{Float32}(Z) ≡ Typ{Float32}(5,5)
            @test AbstractArray{Float32,2}(Z) ≡ Typ{Float32}(5,5)
        end
    end

    @test Fill(1) ≡ Fill{Int}(1) ≡ Fill{Int,0}(1) ≡ Fill{Int,0,Tuple{}}(1,())
    @test Fill(1,(-1,5)) ≡ Fill(1,(0,5))
    @test Fill(1.0,5) isa AbstractVector{Float64}
    @test Fill(1.0,5,5) isa AbstractMatrix{Float64}
    @test Fill(1,5) ≡ Fill(1,(5,))
    @test Fill(1,5,5) ≡ Fill(1,(5,5))
    @test eltype(Fill(1.0,5,5)) == Float64

    @test Matrix{Float64}(Zeros{ComplexF64}(10,10)) == zeros(10,10)
    @test_throws InexactError Matrix{Float64}(Fill(1.0+1.0im,10,10))


    for T in (Int, Float64)
        F = Fill{T, 0}(2)
        @test size(F) == ()
        @test F[] === T(2)

        F = Fill{T}(1, 5)

        @test eltype(F) == T
        @test Array(F) == fill(one(T),5)
        @test Array{T}(F) == fill(one(T),5)
        @test Array{T,1}(F) == fill(one(T),5)

        F = Fill{T}(1, 5, 5)
        @test eltype(F) == T
        @test Array(F) == fill(one(T),5,5)
        @test Array{T}(F) == fill(one(T),5,5)
        @test Array{T,2}(F) == fill(one(T),5,5)

        @test convert(AbstractArray,F) ≡ F
        @test convert(AbstractArray{T},F) ≡ AbstractArray{T}(F) ≡ F
        @test convert(AbstractMatrix{T},F) ≡ AbstractMatrix{T}(F) ≡ F

        @test convert(AbstractArray{Float32},F) ≡ AbstractArray{Float32}(F) ≡
                Fill{Float32}(one(Float32),5,5)
        @test convert(AbstractMatrix{Float32},F) ≡ AbstractMatrix{Float32}(F) ≡
                Fill{Float32}(one(Float32),5,5)

        @test Fill{T}(F) ≡ Fill{T,2}(F) ≡ typeof(F)(F) ≡ F

        show(devnull, MIME("text/plain"), F) # for codecov
    end

    @test Eye(5) isa Diagonal{Float64}
    @test SquareEye(5) isa Diagonal{Float64}
    @test Eye(5) == Eye{Float64}(5) == SquareEye(5) == SquareEye{Float64}(5)
    @test Eye(5,6) == Eye{Float64}(5,6)
    @test Eye(Ones(5,6)) == Eye{Float64}(5,6)
    @test eltype(Eye(5)) == Float64
    @test eltype(Eye(5,6)) == Float64

    @test Eye((Base.OneTo(5),)) ≡ SquareEye((Base.OneTo(5),)) ≡ Eye(5)
    @test Eye((Base.OneTo(5),Base.OneTo(6))) ≡ Eye(5,6)

    for T in (Int, Float64)
        E = Eye{T}(5)
        M = Matrix{T}(I, 5, 5)

        @test eltype(E) == T
        @test Array(E) == M
        @test Array{T}(E) == M
        @test Array{T,2}(E) == M

        @test convert(AbstractArray,E) === E
        @test convert(AbstractArray{T},E) === E
        @test convert(AbstractMatrix{T},E) === E


        @test AbstractArray{Float32}(E) == Eye{Float32}(5)
        @test AbstractArray{Float32}(E) == Eye{Float32}(5, 5)

        @test Eye{T}(randn(4,5)) ≡ Eye{T}(4,5) ≡ Eye{T}((Base.OneTo(4),Base.OneTo(5)))
        @test Eye{T}((Base.OneTo(5),)) ≡ SquareEye{T}((Base.OneTo(5),)) ≡ Eye{T}(5)
    end

    @testset "Bool should change type" begin
        x = Fill(true,5)
        y = x + x
        @test y isa Fill{Int,1}
        @test y[1] == 2

        x = Ones{Bool}(5)
        y = x + x
        @test y isa Fill{Int,1}
        @test y[1] == 2
        @test x + Zeros{Bool}(5) ≡ Ones{Int}(5)
        @test x - Zeros{Bool}(5) ≡ Ones{Int}(5)
        @test Zeros{Bool}(5) + x ≡ Ones{Int}(5)
        @test -x ≡ Fill(-1,5)
    end

    @testset "copy should return Fill" begin
        x = Fill(1.0,10)
        @test copy(x) ≡ x
        x = Zeros(10)
        @test copy(x) ≡ x
        x = Fill([1.,2.],10)
        @test copy(x) == x
        @test copy(x) === x   # because isbits(x)
        @test copy(x) isa Fill
        @test copy(Fill(:a, 4)) == fill(:a, 4)    # FillArrays#63
    end

    @testset "vec" begin
        @test vec(Ones{Int}(5,10)) ≡ Ones{Int}(50)
        @test vec(Zeros{Int}(5,10)) ≡ Zeros{Int}(50)
        @test vec(Zeros{Int}(5,10,20)) ≡ Zeros{Int}(1000)
        @test vec(Fill(1,5,10)) ≡ Fill(1,50)
    end

    @testset "in" begin
        for T in [Zeros, Ones, Fill, Trues, Falses]
            A = T(4, 4)
            @test FillArrays.getindex_value(A) in A
            @test !(FillArrays.getindex_value(A) + 1 in A)
        end
        A = FillArrays.RectDiagonal([1, 2, 3])
        @test 3 in A
        @test 0 in A
        @test !(4 in A)
        A = FillArrays.RectDiagonal([1])
        @test 1 in A
        @test !(0 in A)
        A = FillArrays.RectDiagonal([2], (1:1, 1:4))
        @test 2 in A
        @test 0 in A
        @test !(1 in A)
        @test !(Zeros(1,1) in A)
        A = FillArrays.RectDiagonal(Int[])
        @test !(0 in A)
        A = FillArrays.RectDiagonal(Int[], (1:0, 1:4))
        @test !(0 in A)
    end

    @testset "promotion" begin
        Z = Zeros{Int}(5)
        Zf = Zeros(5)
        @test promote_type(typeof(Z), typeof(Zf)) == typeof(Zf)
        O = Ones{Int}(5)
        Of = Ones{Float64}(5)
        @test promote_type(typeof(O), typeof(Of)) == typeof(Of)
        @test [Z,O] isa Vector{Fill{Int,1,Tuple{Base.OneTo{Int}}}}
        @test [Z,Of] isa Vector{Fill{Float64,1,Tuple{Base.OneTo{Int}}}}
        @test [O,O] isa Vector{Ones{Int,1,Tuple{Base.OneTo{Int}}}}
        @test [O,Of] isa Vector{Ones{Float64,1,Tuple{Base.OneTo{Int}}}}
        @test [Z,Zf] isa Vector{Zeros{Float64,1,Tuple{Base.OneTo{Int}}}}

        @test convert(Ones{Int}, Of) ≡ convert(Ones{Int,1}, Of) ≡ convert(typeof(O), Of) ≡ O
        @test convert(Zeros{Int}, Zf) ≡ convert(Zeros{Int,1}, Zf) ≡ convert(typeof(Z), Zf) ≡ Z

        F = Fill(1, 2)
        Ff = Fill(1.0, 2)
        @test promote_type(typeof(F), typeof(Ff)) == typeof(Ff)

        @test_throws MethodError convert(Zeros{SVector{2,Int}}, Zf)
    end
end

@testset "interface" begin
    struct Twos{T,N} <: FillArrays.AbstractFill{T,N,NTuple{N,Base.OneTo{Int}}}
        sz :: NTuple{N,Int}
    end
    Twos{T}(sz::NTuple{N,Int}) where {T,N} = Twos{T,N}(sz)
    Twos{T}(sz::Vararg{Int,N}) where {T,N} = Twos{T,N}(sz)
    Base.size(A::Twos) = A.sz
    FillArrays.getindex_value(A::Twos{T}) where {T} = oneunit(T) + oneunit(T)

    @testset "broadcasting ambiguities" begin
        A = Twos{Int}(3)
        B = Zeros{Int}(size(A))
        @test A .* B === B
        @test B .* A === B
        @test B ./ A === Zeros{Float64}(size(A))
        @test A .\ B === Zeros{Float64}(size(A))
        @test A ./ B === Fill(Inf, size(A))
    end
end

@testset "isassigned" begin
    for f in (Fill("", 3, 4), Zeros(3,4), Ones(3,4))
        @test !isassigned(f, 0, 0)
        @test isassigned(f, 2, 2)
        @test !isassigned(f, 10, 10)
        @test_throws ArgumentError isassigned(f, true)
    end
end

@testset "indexing" begin
    A = Fill(3.0,5)
    @test A[1:3] ≡ Fill(3.0,3)
    @test A[1:3,1:1] ≡ Fill(3.0,3,1)
    @test_throws BoundsError A[1:3,2]
    @test_throws BoundsError A[1:26]
    @test A[[true, false, true, false, false]] ≡ Fill(3.0, 2)
    A = Fill(3.0, 2, 2)
    @test A[[true true; true false]] ≡ Fill(3.0, 3)
    @test_throws BoundsError A[[true, false]]

    A = Ones{Int}(5,5)
    @test A[1:3] ≡ Ones{Int}(3)
    @test A[1:3,1:2] ≡ Ones{Int}(3,2)
    @test A[1:3,2] ≡ Ones{Int}(3)
    @test_throws BoundsError A[1:26]
    A = Ones{Int}(2,2)
    @test A[[true false; true false]] ≡ Ones{Int}(2)
    @test A[[true, false, true, false]] ≡ Ones{Int}(2)
    @test_throws BoundsError A[[true false false; true false false]]

    A = Zeros{Int}(5,5)
    @test A[1:3] ≡ Zeros{Int}(3)
    @test A[1:3,1:2] ≡ Zeros{Int}(3,2)
    @test A[1:3,2] ≡ Zeros{Int}(3)
    @test_throws BoundsError A[1:26]
    A = Zeros{Int}(2,2)
    @test A[[true false; true false]] ≡ Zeros{Int}(2)
    @test A[[true, false, true, false]] ≡ Zeros{Int}(2)
    @test_throws BoundsError A[[true false false; true false false]]

    @testset "colon" begin
        @test Ones(2)[:] ≡ Ones(2)[Base.Slice(Base.OneTo(2))] ≡ Ones(2)
        @test Zeros(2)[:] ≡ Zeros(2)[Base.Slice(Base.OneTo(2))] ≡ Zeros(2)
        @test Fill(3.0,2)[:] ≡ Fill(3.0,2)[Base.Slice(Base.OneTo(2))] ≡ Fill(3.0,2)

        @test Ones(2,2)[:,:] ≡ Ones(2,2)[Base.Slice(Base.OneTo(2)),Base.Slice(Base.OneTo(2))] ≡ Ones(2,2)
        @test Zeros(2,2)[:,:] ≡ Zeros(2)[Base.Slice(Base.OneTo(2)),Base.Slice(Base.OneTo(2))] ≡ Zeros(2,2)
        @test Fill(3.0,2,2)[:,:] ≡ Fill(3.0,2,2)[Base.Slice(Base.OneTo(2)),Base.Slice(Base.OneTo(2))] ≡ Fill(3.0,2,2)
    end

    @testset "mixed integer / vector /colon" begin
        a = Fill(2.0,5)
        z = Zeros(5)
        @test a[1:5] ≡ a[:] ≡ a
        @test z[1:5] ≡ z[:] ≡ z

        A = Fill(2.0,5,6)
        Z = Zeros(5,6)
        @test A[:,1] ≡ A[1:5,1] ≡ Fill(2.0,5)
        @test A[1,:] ≡ A[1,1:6] ≡ Fill(2.0,6)
        @test A[:,:] ≡ A[1:5,1:6] ≡ A[1:5,:] ≡ A[:,1:6] ≡ A
        @test Z[:,1] ≡ Z[1:5,1] ≡ Zeros(5)
        @test Z[1,:] ≡ Z[1,1:6] ≡ Zeros(6)
        @test Z[:,:] ≡ Z[1:5,1:6] ≡ Z[1:5,:] ≡ Z[:,1:6] ≡ Z

        A = Fill(2.0,5,6,7)
        Z = Zeros(5,6,7)
        @test A[:,1,1] ≡ A[1:5,1,1] ≡ Fill(2.0,5)
        @test A[1,:,1] ≡ A[1,1:6,1] ≡ Fill(2.0,6)
        @test A[:,:,:] ≡ A[1:5,1:6,1:7] ≡ A[1:5,:,1:7] ≡ A[:,1:6,1:7] ≡ A
    end

    @testset "StepRangeLen convert" begin
        for (z,s) in  ((Zeros{Int}(5), StepRangeLen(0, 0, 5)), (Ones{Int}(5), StepRangeLen(1, 0, 5)), (Fill(2,5), StepRangeLen(2, 0, 5)))
            @test s == z
            @test StepRangeLen(z) ≡ convert(StepRangeLen, z) ≡ convert(StepRangeLen{Int}, z) ≡ convert(typeof(s), z) ≡ convert(AbstractRange, z) ≡ s
        end
    end
end

@testset "RectDiagonal" begin
    data = 1:3
    expected_size = (5, 3)
    expected_axes = Base.OneTo.(expected_size)
    expected_matrix = [1 0 0; 0 2 0; 0 0 3; 0 0 0; 0 0 0]
    expected = RectDiagonal{Int, UnitRange{Int}}(data, expected_axes)

    @test axes(expected) == expected_axes
    @test size(expected) == expected_size
    @test (axes(expected, 1), axes(expected, 2)) == expected_axes
    @test (size(expected, 1), size(expected, 2)) == expected_size

    @test expected == expected_matrix
    @test Matrix(expected) == expected_matrix
    @test expected[:, 2] == expected_matrix[:, 2]
    @test expected[2, :] == expected_matrix[2, :]
    @test expected[5, :] == expected_matrix[5, :]


    for Typ in (RectDiagonal, RectDiagonal{Int}, RectDiagonal{Int, UnitRange{Int}})
        @test Typ(data) == expected[1:3, 1:3]
        @test Typ(data, expected_axes) == expected
        @test Typ(data, expected_axes...) == expected
        @test Typ(data, expected_size) == expected
        @test Typ(data, expected_size...) == expected
    end

    @test diag(expected) === expected.diag

    mut = RectDiagonal(collect(data), expected_axes)
    @test mut == expected
    @test mut == expected_matrix
    mut[1, 1] = 5
    @test mut[1] == 5
    @test diag(mut) == [5, 2, 3]
    mut[2, 1] = 0
    @test_throws ArgumentError mut[2, 1] = 9

    D = RectDiagonal([1.,2.], (Base.OneTo(3),Base.OneTo(2)))
    @test stringmime("text/plain", D) == "3×2 RectDiagonal{Float64, Vector{Float64}, Tuple{Base.OneTo{$Int}, Base.OneTo{$Int}}}:\n 1.0   ⋅ \n  ⋅   2.0\n  ⋅    ⋅ "
end

# Check that all pair-wise combinations of + / - elements of As and Bs yield the correct
# type, and produce numerically correct results.
as_array(x::AbstractArray) = Array(x)
as_array(x::UniformScaling) = x
isapprox_or_undef(a::Number, b::Number) = (a ≈ b) || isequal(a, b)
isapprox_or_undef(a, b) = all(((x,y),) -> isapprox_or_undef(x,y), zip(a, b))
function test_addition_subtraction_dot(As, Bs, Tout::Type)
    for A in As, B in Bs
        @testset "$(typeof(A)) and $(typeof(B))" begin
            @test @inferred(A + B) isa Tout{promote_type(eltype(A), eltype(B))}
            @test isapprox_or_undef(as_array(A + B), as_array(A) + as_array(B))

            @test @inferred(A - B) isa Tout{promote_type(eltype(A), eltype(B))}
            @test isapprox_or_undef(as_array(A - B), as_array(A) - as_array(B))

            @test @inferred(B + A) isa Tout{promote_type(eltype(B), eltype(A))}
            @test isapprox_or_undef(as_array(B + A), as_array(B) + as_array(A))

            @test @inferred(B - A) isa Tout{promote_type(eltype(B), eltype(A))}
            @test isapprox_or_undef(as_array(B - A), as_array(B) - as_array(A))

            # Julia 1.6 doesn't support dot(UniformScaling)
            if VERSION < v"1.6.0" || VERSION >= v"1.8.0"
                d1 = dot(A, B)
                d2 = dot(as_array(A), as_array(B))
                d3 = dot(B, A)
                d4 = dot(as_array(B), as_array(A))
                @test d1 ≈ d2 || d1 ≡ d2
                @test d3 ≈ d4 || d3 ≡ d4
            end
        end
    end
end

# Check that all permutations of + / - throw a `DimensionMismatch` exception.
function test_addition_and_subtraction_dim_mismatch(a, b)
    @testset "$(typeof(a)) ± $(typeof(b))" begin
        @test_throws DimensionMismatch a + b
        @test_throws DimensionMismatch a - b
        @test_throws DimensionMismatch b + a
        @test_throws DimensionMismatch b - a
    end
end

@testset "FillArray addition and subtraction" begin
    test_addition_and_subtraction_dim_mismatch(Zeros(5), Zeros(6))
    test_addition_and_subtraction_dim_mismatch(Zeros(5), Zeros{Int}(6))
    test_addition_and_subtraction_dim_mismatch(Zeros(5), Zeros(6,6))
    test_addition_and_subtraction_dim_mismatch(Zeros(5), Zeros{Int}(6,5))

    # Construct FillArray for repeated use.
    rng = MersenneTwister(123456)
    A_fill, B_fill = Fill(randn(rng, Float64), 5), Fill(4, 5)

    # Unary +/- constructs a new FillArray.
    @test +A_fill === A_fill
    @test -A_fill === Fill(-A_fill.value, 5)

    # FillArray +/- FillArray should construct a new FillArray.
    test_addition_subtraction_dot((A_fill, B_fill), (A_fill, B_fill), Fill)
    test_addition_and_subtraction_dim_mismatch(A_fill, Fill(randn(rng), 5, 2))

    # FillArray + Array (etc) should construct a new Array using `getindex`.
    B_dense = (randn(rng, 5), [5, 4, 3, 2, 1], fill(Inf, 5), fill(NaN, 5))
    test_addition_subtraction_dot((A_fill, B_fill), B_dense, Array)
    test_addition_and_subtraction_dim_mismatch(A_fill, randn(rng, 5, 2))

    # FillArray + StepLenRange / UnitRange (etc) should yield an AbstractRange.
    A_ur, B_ur = 1.0:5.0, 6:10
    test_addition_subtraction_dot((A_fill, B_fill), (A_ur, B_ur), AbstractRange)
    test_addition_and_subtraction_dim_mismatch(A_fill, 1.0:6.0)
    test_addition_and_subtraction_dim_mismatch(A_fill, 5:10)

    # FillArray + UniformScaling should yield a Matrix in general
    As_fill_square = (Fill(randn(rng, Float64), 3, 3), Fill(5, 4, 4))
    Bs_us = (UniformScaling(2.3), UniformScaling(3))
    test_addition_subtraction_dot(As_fill_square, Bs_us, Matrix)
    As_fill_nonsquare = (Fill(randn(rng, Float64), 3, 2), Fill(5, 3, 4))
    for A in As_fill_nonsquare, B in Bs_us
        test_addition_and_subtraction_dim_mismatch(A, B)
    end

    # FillArray + StaticArray should not have ambiguities
    A_svec, B_svec = SVector{5}(rand(5)), SVector(1, 2, 3, 4, 5)
    test_addition_subtraction_dot((A_fill, B_fill, Zeros(5)), (A_svec, B_svec), SVector{5})

    # Issue #224
    A_matmat, B_matmat = Fill(rand(3,3),5), [rand(3,3) for n=1:5]
    test_addition_subtraction_dot((A_matmat,), (A_matmat,), Fill)
    test_addition_subtraction_dot((B_matmat,), (A_matmat,), Vector)

    # Optimizations for Zeros and RectOrDiagonal{<:Any, <:AbstractFill}
    As_special_square = (
        Zeros(3, 3), Zeros{Int}(4, 4),
        Eye(3), Eye{Int}(4), Eye(3, 3), Eye{Int}(4, 4),
        Diagonal(Fill(randn(rng, Float64), 3)), Diagonal(Fill(3, 4)),
        RectDiagonal(Fill(randn(rng, Float64), 3), 3, 3), RectDiagonal(Fill(3, 4), 4, 4)
    )
    DiagonalAbstractFill{T} = Diagonal{T, <:AbstractFill{T, 1}}
    test_addition_subtraction_dot(As_special_square, Bs_us, DiagonalAbstractFill)
    As_special_nonsquare = (
        Zeros(3, 2), Zeros{Int}(3, 4),
        Eye(3, 2), Eye{Int}(3, 4),
        RectDiagonal(Fill(randn(rng, Float64), 2), 3, 2), RectDiagonal(Fill(3, 3), 3, 4)
    )
    for A in As_special_nonsquare, B in Bs_us
        test_addition_and_subtraction_dim_mismatch(A, B)
    end

    @testset "Zeros" begin
        As = ([1,2], Float64[1,2], Int8[1,2], ComplexF16[2,4])
        Zs = (TZ -> Zeros{TZ}(2)).((Int, Float64, Int8, ComplexF64))
        test_addition_subtraction_dot(As, Zs, Vector)
        for A in As, Z in (TZ -> Zeros{TZ}(3)).((Int, Float64, Int8, ComplexF64))
            test_addition_and_subtraction_dim_mismatch(A, Z)
        end

        As = (@SArray([1,2]), @SArray(Float64[1,2]), @SArray(Int8[1,2]), @SArray(ComplexF16[2,4]))
        test_addition_subtraction_dot(As, Zs, SVector{2})
        for A in As, Z in (TZ -> Zeros{TZ}(3)).((Int, Float64, Int8, ComplexF64))
            test_addition_and_subtraction_dim_mismatch(A, Z)
        end
    end
end

@testset "Other matrix types" begin
    @test Diagonal(Zeros(5)) == Diagonal(zeros(5))

    @test Diagonal(Zeros(8,5)) == Diagonal(zeros(5))
    @test convert(Diagonal, Zeros(5,5)) == Diagonal(zeros(5))
    @test_throws DimensionMismatch convert(Diagonal, Zeros(8,5))

    @test convert(Diagonal{Int}, Zeros(5,5)) == Diagonal(zeros(Int,5))
    @test_throws DimensionMismatch convert(Diagonal{Int}, Zeros(8,5))


    @test Diagonal(Eye(8,5)) == Diagonal(ones(5))
    @test convert(Diagonal, Eye(5)) == Diagonal(ones(5))
    @test convert(Diagonal{Int}, Eye(5)) == Diagonal(ones(Int,5))

    for E in (Eye(2,4), Eye(3))
        M = collect(E)
        for i in -5:5
            @test diag(E, i) == diag(M, i)
        end
    end
end

@testset "one" begin
    @testset for A in Any[Eye(4), Zeros(4,4), Ones(4,4), Fill(3,4,4)]
        B = one(A)
        @test B * A == A * B == A
    end
    @test_throws ArgumentError one(Ones(3,4))
    @test_throws ArgumentError one(Ones((3:5,4:5)))
end

@testset "Sparse vectors and matrices" begin
    @test SparseVector(Zeros(5)) ==
            SparseVector{Int}(Zeros(5)) ==
            SparseVector{Float64}(Zeros(5)) ==
            SparseVector{Float64,Int}(Zeros(5)) ==
            convert(AbstractSparseArray,Zeros(5)) ==
            convert(AbstractSparseVector,Zeros(5)) ==
            convert(AbstractSparseArray{Float64},Zeros(5)) ==
            convert(AbstractSparseVector{Float64},Zeros(5)) ==
            convert(AbstractSparseVector{Float64,Int},Zeros(5)) ==
            spzeros(5)

    for (Mat, SMat) in ((Zeros(5,5), spzeros(5,5)), (Zeros(6,5), spzeros(6,5)),
                        (Eye(5), sparse(I,5,5)), (Eye(6,5), sparse(I,6,5)))
        @test SparseMatrixCSC(Mat) ==
                SparseMatrixCSC{Int}(Mat) ==
                SparseMatrixCSC{Float64}(Mat) ==
                SparseMatrixCSC{Float64,Int}(Mat) ==
                convert(AbstractSparseArray,Mat) ==
                convert(AbstractSparseMatrix,Mat) ==
                convert(AbstractSparseArray{Float64},Mat) ==
                convert(AbstractSparseArray{Float64,Int},Mat) ==
                convert(AbstractSparseMatrix{Float64},Mat) ==
                convert(AbstractSparseMatrix{Float64,Int},Mat) ==
                SMat
    end

    function testsparsediag(E)
        S = @inferred SparseMatrixCSC(E)
        @test S == E
        S = @inferred SparseMatrixCSC{Float64}(E)
        @test S == E
        @test S isa SparseMatrixCSC{Float64}
        @test convert(SparseMatrixCSC{Float64}, E) == S
        S = @inferred SparseMatrixCSC{Float64,Int32}(E)
        @test S == E
        @test S isa SparseMatrixCSC{Float64,Int32}
        @test convert(SparseMatrixCSC{Float64,Int32}, E) == S
    end

    for f in (Fill(Int8(4),3), Ones{Int8}(3), Zeros{Int8}(3))
        E = Diagonal(f)
        testsparsediag(E)
        for sz in ((3,6), (6,3), (3,3))
            E = RectDiagonal(f, sz)
            testsparsediag(E)
        end
    end
end

@testset "==" begin
    @test Zeros(5,4) == Fill(0,5,4)
    @test Zeros(5,4) ≠ Zeros(3)
    @test Ones(5,4) == Fill(1,5,4)
end

@testset "Rank" begin
    @test rank(Zeros(5,4)) == 0
    @test rank(Ones(5,4)) == 1
    @test rank(Fill(2,5,4)) == 1
    @test rank(Fill(0,5,4)) == 0
    @test rank(Eye(2)) == 2
end

@testset "ishermitian" begin
    for el in (2, 3+0im, 4+5im), size in [(3,3), (3,4)]
        @test issymmetric(Fill(el, size...)) == issymmetric(fill(el, size...))
        @test ishermitian(Fill(el, size...)) == ishermitian(fill(el, size...))
    end
end

@testset "BigInt indices" begin
    for A in (Zeros(BigInt(100)), Ones(BigInt(100)), Fill(2, BigInt(100)))
        @test length(A) isa BigInt
        @test axes(A) == tuple(Base.OneTo{BigInt}(BigInt(100)))
        @test size(A) isa Tuple{BigInt}
    end
    for A in (Eye(BigInt(100)), Eye(BigInt(100), BigInt(100)))
        @test length(A) isa BigInt
        @test axes(A) == tuple(Base.OneTo{BigInt}(BigInt(100)),Base.OneTo{BigInt}(BigInt(100)))
        @test size(A) isa Tuple{BigInt,BigInt}
    end
    for A in (Zeros(BigInt(10), 10), Ones(BigInt(10), 10), Fill(2.0, (BigInt(10), 10)), Eye(BigInt(10), 8))
        @test size(A) isa Tuple{BigInt,Int}
    end

end

@testset "IndexStyle" begin
    @test IndexStyle(Zeros(5,5)) == IndexStyle(typeof(Zeros(5,5))) == IndexLinear()
end

@testset "Identities" begin
    @test Zeros(3,4) * randn(4,5) === randn(3,4) * Zeros(4,5) === Zeros(3, 5)
    @test_throws DimensionMismatch randn(3,4) * Zeros(3, 3)
    @test eltype(Zeros{Int}(3,4) * fill(1, 4, 5)) == Int
    @test eltype(Zeros{Int}(3,4) * fill(3.4, 4, 5)) == Float64
    @test Zeros(3, 4) * randn(4) == Zeros(3, 4) * Zeros(4) == Zeros(3)
    @test Zeros(3, 4) * Zeros(4, 5) === Zeros(3, 5)

    @test_throws MethodError [1,2,3]*Zeros(1) # Not defined for [1,2,3]*[0] either
    @test [1,2,3]*Zeros(1,3) ≡ Zeros(3,3)
    @test_throws MethodError [1,2,3]*Zeros(3) # Not defined for [1,2,3]*[0,0,0] either

    @testset "Matrix multiplication with array elements" begin
        x = [1 2; 3 4]
        z = zero(SVector{2,Int})
        ZV = Zeros{typeof(z)}(2)
        A = Fill(x, 3, 2) * ZV
        @test A isa Fill
        @test size(A) == (3,)
        @test A[1] == x * z
        @test_throws DimensionMismatch Fill(x, 1, 1) * ZV
        @test_throws DimensionMismatch Fill(oneton(1,1), 1, length(ZV)) * ZV

        @test_throws DimensionMismatch Ones(SMatrix{3,3,Int,9},2) * Ones(SMatrix{2,2,Int,4},1,2)
    end

    @testset "Check multiplication by Adjoint vectors works as expected." begin
        @test @inferred(oneton(4, 3)' * Zeros(4)) ≡ Zeros(3)
        @test @inferred(oneton(4)' * Zeros(4)) ≡ @inferred(transpose(oneton(4)) * Zeros(4)) == 0.0
        @test [1, 2, 3]' * Zeros{Int}(3) ≡ zero(Int)
        @test [SVector(1,2)', SVector(2,3)', SVector(3,4)']' * Zeros{Int}(3) === SVector(0,0)
        @test_throws DimensionMismatch oneton(4)' * Zeros(3)
        @test Zeros(5)' * oneton(5,3) ≡ Zeros(5)'*Zeros(5,3) ≡ Zeros(5)'*Ones(5,3) ≡ Zeros(3)'
        @test abs(Zeros(5)' * oneton(5)) == abs(Zeros(5)' * Zeros(5)) ≡ abs(Zeros(5)' * Ones(5)) == 0.0
        @test Zeros(5) * Zeros(6)' ≡ Zeros(5,1) * Zeros(6)' ≡ Zeros(5,6)
        @test oneton(5) * Zeros(6)' ≡ oneton(5,1) * Zeros(6)' ≡ Zeros(5,6)
        @test Zeros(5) * oneton(6)' ≡ Zeros(5,6)

        @test @inferred(Zeros{Int}(0)' * Zeros{Int}(0)) === zero(Int)

        @test Any[1,2.0]' * Zeros{Int}(2) == 0
        @test Real[1,2.0]' * Zeros{Int}(2) == 0

        @test @inferred(([[1,2]])' * Zeros{SVector{2,Int}}(1)) ≡ 0
        @test ([[1,2], [1,2]])' * Zeros{SVector{2,Int}}(2) ≡ 0
        @test_throws DimensionMismatch ([[1,2,3]])' * Zeros{SVector{2,Int}}(1)
        @test_throws DimensionMismatch ([[1,2,3], [1,2]])' * Zeros{SVector{2,Int}}(2)

        A = SMatrix{2,1,Int,2}[]'
        B = Zeros(SVector{2,Int},0)
        C = collect(B)
        @test @inferred(A * B) == @inferred(A * C)
    end

    @testset "Check multiplication by Transpose-d vectors works as expected." begin
        @test transpose(oneton(4, 3)) * Zeros(4) === Zeros(3)
        @test transpose(oneton(4)) * Zeros(4) == 0.0
        @test transpose([1, 2, 3]) * Zeros{Int}(3) === zero(Int)
        @test_throws DimensionMismatch transpose(oneton(4)) * Zeros(3)
        @test transpose(Zeros(5)) * oneton(5,3) ≡ transpose(Zeros(5))*Zeros(5,3) ≡ transpose(Zeros(5))*Ones(5,3) ≡ transpose(Zeros(3))
        @test abs(transpose(Zeros(5)) * oneton(5)) ≡ abs(transpose(Zeros(5)) * Zeros(5)) ≡ abs(transpose(Zeros(5)) * Ones(5)) ≡ 0.0
        @test oneton(5) * transpose(Zeros(6)) ≡ oneton(5,1) * transpose(Zeros(6)) ≡ Zeros(5,6)
        @test Zeros(5) * transpose(oneton(6)) ≡ Zeros(5,6)
        @test transpose(oneton(5)) * Zeros(5) == 0.0
        @test transpose(oneton(5) .+ im) * Zeros(5) == 0.0 + 0im

        @test @inferred(transpose(Zeros{Int}(0)) * Zeros{Int}(0)) === zero(Int)

        @test transpose(Any[1,2.0]) * Zeros{Int}(2) == 0
        @test transpose(Real[1,2.0]) * Zeros{Int}(2) == 0

        @test @inferred(transpose([[1,2]]) * Zeros{SVector{2,Int}}(1)) ≡ 0
        @test transpose([[1,2], [1,2]]) * Zeros{SVector{2,Int}}(2) ≡ 0
        @test_throws DimensionMismatch transpose([[1,2,3]]) * Zeros{SVector{2,Int}}(1)
        @test_throws DimensionMismatch transpose([[1,2,3], [1,2]]) * Zeros{SVector{2,Int}}(2)

        A = transpose(SMatrix{2,1,Int,2}[])
        B = Zeros(SVector{2,Int},0)
        C = collect(B)
        @test @inferred(A * B) == @inferred(A * C)

        @testset "Diagonal mul introduced in v1.9" begin
            @test Zeros(5)'*Diagonal(1:5) ≡ Zeros(5)'
            @test transpose(Zeros(5))*Diagonal(1:5) ≡ transpose(Zeros(5))
            @test Zeros(5)'*Diagonal(1:5)*(1:5) ==
                (1:5)'*Diagonal(1:5)*Zeros(5) ==
                transpose(1:5)*Diagonal(1:5)*Zeros(5) ==
                Zeros(5)'*Diagonal(1:5)*Zeros(5) ==
                transpose(Zeros(5))*Diagonal(1:5)*Zeros(5) ==
                transpose(Zeros(5))*Diagonal(1:5)*(1:5)

            @test_throws DimensionMismatch Zeros(6)'*Diagonal(1:5)*Zeros(5)
            @test_throws DimensionMismatch Zeros(5)'*Diagonal(1:6)*Zeros(5)
            @test_throws DimensionMismatch Zeros(5)'*Diagonal(1:5)*Zeros(6)
        end
    end

    z1, z2 = Zeros{Float64}(4), Zeros{Int}(4)

    @testset "`Zeros` are closed under addition and subtraction (both unary and binary)." begin
        @test +(Zeros{Float64}(3, 5)) === Zeros{Float64}(3, 5)
        @test -(Zeros{Float32}(5, 2)) === Zeros{Float32}(5, 2)

        @test +(z1) === z1
        @test -(z1) === z1

        test_addition_subtraction_dot((z1, z2), (z1, z2), Zeros)
        test_addition_and_subtraction_dim_mismatch(z1, Zeros{Float64}(4, 2))
    end

    # `Zeros` +/- `Fill`s should yield `Fills`.
    fill1, fill2 = Fill(5.0, 4), Fill(5, 4)
    test_addition_subtraction_dot((z1, z2), (fill1, fill2), Fill)
    test_addition_and_subtraction_dim_mismatch(z1, Fill(5, 5))

    X = randn(3, 5)
    for op in [+, -]

        # Addition / subtraction with same eltypes.
        @test op(Zeros(6, 4), Zeros(6, 4)) === Zeros(6, 4)
        @test_throws DimensionMismatch op(X, Zeros(4, 6))
        @test eltype(op(Zeros(3, 5), X)) == Float64

        # Different eltypes, the other way around.
        @test op(X, Zeros{Float32}(3, 5)) isa Matrix{Float64}
        @test !(op(X, Zeros{Float32}(3, 5)) === X)
        @test op(X, Zeros{Float32}(3, 5)) == X
        @test !(op(X, Zeros{ComplexF64}(3, 5)) === X)
        @test op(X, Zeros{ComplexF64}(3, 5)) == X

        # Addition / subtraction of Zeros.
        @test eltype(op(Zeros{Float64}(4, 5), Zeros{Int}(4, 5))) == Float64
        @test eltype(op(Zeros{Int}(5, 4), Zeros{Float32}(5, 4))) == Float32
        @test op(Zeros{Float64}(4, 5), Zeros{Int}(4, 5)) isa Zeros{Float64}
        @test op(Zeros{Float64}(4, 5), Zeros{Int}(4, 5)) === Zeros{Float64}(4, 5)
    end

    @testset "Zeros +/- dense where + / - have different results." begin
        @test +(Zeros(3, 5), X) == X && +(X, Zeros(3, 5)) == X
        @test !(Zeros(3, 5) + X === X) && !(X + Zeros(3, 5) === X)
        @test -(Zeros(3, 5), X) == -X
    end

    @testset "Addition with different eltypes." begin
        @test +(Zeros{Float32}(3, 5), X) isa Matrix{Float64}
        @test !(+(Zeros{Float32}(3, 5), X) === X)
        @test +(Zeros{Float32}(3, 5), X) == X
        @test !(+(Zeros{ComplexF64}(3, 5), X) === X)
        @test +(Zeros{ComplexF64}(3, 5), X) == X
    end

    @testset "Subtraction with different eltypes." begin
        @test -(Zeros{Float32}(3, 5), X) isa Matrix{Float64}
        @test -(Zeros{Float32}(3, 5), X) == -X
        @test -(Zeros{ComplexF64}(3, 5), X) == -X
    end

    @testset "Tests for ranges." begin
        X = randn(5)
        @test !(Zeros(5) + X ≡ X)
        @test Zeros{Int}(5) + (1:5) ≡ (1:5) + Zeros{Int}(5) ≡ (1:5)
        @test Zeros(5) + (1:5) ≡ (1:5) + Zeros(5) ≡ (1.0:1.0:5.0)
        @test (1:5) - Zeros{Int}(5) ≡ (1:5)
        @test Zeros{Int}(5) - (1:5) ≡ -1:-1:-5
        @test Zeros(5) - (1:5) ≡ -1.0:-1.0:-5.0
        @test Zeros{Int}(5) + (1.0:5) ≡ (1.0:5) + Zeros{Int}(5) ≡ 1.0:5
    end

    @testset "test Base.zero" begin
        @test zero(Zeros(10)) == Zeros(10)
        @test zero(Ones(10,10)) == Zeros(10,10)
        @test zero(Fill(0.5, 10, 10)) == Zeros(10,10)
    end

    @testset "Matrix ±" begin
        x = Fill([1,2], 5)
        z = Zeros{SVector{2,Int}}(5)
        @test +(z) ≡ -(z) ≡ z
        @test +(x) == x
        @test -(x) == Fill(-[1,2], 5)
    end
end

@testset "maximum/minimum/svd/sort" begin
    @test maximum(Fill(1, 1_000_000_000)) == minimum(Fill(1, 1_000_000_000)) == 1
    @test svdvals(fill(2,5,6)) ≈ svdvals(Fill(2,5,6))
    @test svdvals(Eye(5)) === Fill(1.0,5)
    @test sort(Ones(5)) == sort!(Ones(5))

    @test_throws MethodError issorted(Fill(im, 2))
    @test_throws MethodError sort(Fill(im, 2))
    @test_throws MethodError sort!(Fill(im, 2))
end

@testset "Cumsum, accumulate and diff" begin
    @test @inferred(sum(Fill(3,10))) ≡ 30
    @test @inferred(reduce(+, Fill(3,10))) ≡ 30
    @test @inferred(sum(x -> x + 1, Fill(3,10))) ≡ 40
    @test @inferred(cumsum(Fill(3,10))) ≡ @inferred(accumulate(+, Fill(3,10))) ≡ StepRangeLen(3,3,10)
    @test @inferred(accumulate(-, Fill(3,10))) ≡ StepRangeLen(3,-3,10)

    @test @inferred(sum(Ones(10))) ≡ 10.0
    @test @inferred(sum(x -> x + 1, Ones(10))) ≡ 20.0
    @test @inferred(cumsum(Ones(10))) ≡ @inferred(accumulate(+, Ones(10))) ≡ StepRangeLen(1.0, 1.0, 10)
    @test @inferred(accumulate(-, Ones(10))) ≡ StepRangeLen(1.0,-1.0,10)

    @test sum(Ones{Int}(10)) ≡ 10
    @test sum(x -> x + 1, Ones{Int}(10)) ≡ 20
    @test cumsum(Ones{Int}(10)) ≡ accumulate(+,Ones{Int}(10)) ≡ Base.OneTo(10)
    @test accumulate(-, Ones{Int}(10)) ≡ StepRangeLen(1,-1,10)

    @test sum(Zeros(10)) ≡ 0.0
    @test sum(x -> x + 1, Zeros(10)) ≡ 10.0
    @test cumsum(Zeros(10)) ≡ accumulate(+,Zeros(10)) ≡ accumulate(-,Zeros(10)) ≡ Zeros(10)

    @test sum(Zeros{Int}(10)) ≡ 0
    @test sum(x -> x + 1, Zeros{Int}(10)) ≡ 10
    @test cumsum(Zeros{Int}(10)) ≡ accumulate(+,Zeros{Int}(10)) ≡ accumulate(-,Zeros{Int}(10)) ≡ Zeros{Int}(10)

    # we want cumsum of fills to match the types of the standard cusum
    @test all(cumsum(Zeros{Bool}(10)) .≡ cumsum(zeros(Bool,10)))
    @test all(accumulate(+, Zeros{Bool}(10)) .≡ accumulate(+, zeros(Bool,10)) .≡ accumulate(-, zeros(Bool,10)))
    @test cumsum(Zeros{Bool}(10)) ≡ accumulate(+, Zeros{Bool}(10)) ≡ accumulate(-, Zeros{Bool}(10)) ≡ Zeros{Int}(10)
    @test cumsum(Ones{Bool}(10)) ≡ accumulate(+, Ones{Bool}(10)) ≡ Base.OneTo{Int}(10)
    @test all(cumsum(Fill(true,10)) .≡ cumsum(fill(true,10)))
    @test cumsum(Fill(true,10)) ≡ StepRangeLen(1, true, 10)

    @test all(cumsum(Zeros{UInt8}(10)) .≡ cumsum(zeros(UInt8,10)))
    @test all(accumulate(+, Zeros{UInt8}(10)) .≡ accumulate(+, zeros(UInt8,10)))
    @test cumsum(Zeros{UInt8}(10)) ≡ Zeros{UInt64}(10)
    @test accumulate(+, Zeros{UInt8}(10)) ≡ accumulate(-, Zeros{UInt8}(10)) ≡ Zeros{UInt8}(10)
    
    @test all(cumsum(Ones{UInt8}(10)) .≡ cumsum(ones(UInt8,10)))
    @test all(accumulate(+, Ones{UInt8}(10)) .≡ accumulate(+, ones(UInt8,10)))
    @test cumsum(Ones{UInt8}(10)) ≡ Base.OneTo(UInt64(10))
    @test accumulate(+, Ones{UInt8}(10)) ≡ Base.OneTo(UInt8(10))
    
    @test all(cumsum(Fill(UInt8(2),10)) .≡ cumsum(fill(UInt8(2),10)))
    @test all(accumulate(+,  Fill(UInt8(2))) .≡ accumulate(+, fill(UInt8(2))))
    @test cumsum(Fill(UInt8(2),10)) ≡ StepRangeLen(UInt64(2), UInt8(2), 10)
    @test accumulate(+, Fill(UInt8(2),10)) ≡ StepRangeLen(UInt8(2), UInt8(2), 10)

    @test diff(Fill(1,10)) ≡ Zeros{Int}(9)
    @test diff(Ones{Float64}(10)) ≡ Zeros{Float64}(9)
    @test_throws UndefKeywordError cumsum(Fill(1,1,5))

    @test @inferred(sum([Ones(4)])) ≡ Fill(1.0, 4)
    @test @inferred(sum([Trues(4)])) ≡ Fill(1, 4)

    @testset "infinite arrays" begin
        r = InfiniteArrays.OneToInf()
        A = Ones{Int}((r,))
        @test isinf(sum(A))
        @test sum(A) == length(A)
        @test sum(x->x^2, A) == sum(A.^2)
        @testset "IteratorSize" begin
            @test (@inferred Base.IteratorSize(Ones())) == Base.IteratorSize(ones())
            @test (@inferred Base.IteratorSize(Ones(2))) == Base.IteratorSize(ones(2))
            @test (@inferred Base.IteratorSize(Ones(r))) == Base.IsInfinite()
            @test (@inferred Base.IteratorSize(Fill(2, (1:2, 1:2)))) == Base.HasShape{2}()
            @test (@inferred Base.IteratorSize(Fill(2, (1:2, r)))) == Base.IsInfinite()
            @test (@inferred Base.IteratorSize(Fill(2, (r, 1:2)))) == Base.IsInfinite()
            @test (@inferred Base.IteratorSize(Fill(2, (r, r)))) == Base.IsInfinite()
        end

        @test issorted(Fill(2, (InfiniteArrays.OneToInf(),)))
    end
end

@testset "Broadcast" begin
    x = Fill(5,5)
    @test (.+)(x) ≡ x
    @test (.-)(x) ≡ -x
    @test exp.(x) ≡ Fill(exp(5),5)
    @test x .+ 1 ≡ Fill(6,5)
    @test 1 .+ x ≡ Fill(6,5)
    @test x .+ x ≡ Fill(10,5)
    @test x .+ Ones(5) ≡ Fill(6.0,5)
    f = (x,y) -> cos(x*y)
    @test f.(x, Ones(5)) ≡ Fill(f(5,1.0),5)
    @test x .^ 1 ≡ Fill(5,5)

    y = Ones(5,5)
    @test (.+)(y) ≡ Ones(5,5)
    @test (.-)(y) ≡ Fill(-1.0,5,5)
    @test exp.(y) ≡ Fill(exp(1),5,5)
    @test y .+ 1 ≡ Fill(2.0,5,5)
    @test y .+ y ≡ Fill(2.0,5,5)
    @test y .* y ≡ y ./ y ≡ y .\ y ≡ y
    @test y .^ 1 ≡ y .^ 0 ≡ Ones(5,5)

    rng = MersenneTwister(123456)
    sizes = [(5, 4), (5, 1), (1, 4), (1, 1), (5,)]
    for sx in sizes, sy in sizes
        x, y = Fill(randn(rng), sx), Fill(randn(rng), sy)
        x_one, y_one = Ones(sx), Ones(sy)
        x_zero, y_zero = Zeros(sx), Zeros(sy)
        x_dense, y_dense = randn(rng, sx), randn(rng, sy)

        for x in [x, x_one, x_zero, x_dense], y in [y, y_one, y_zero, y_dense]
            @test x .+ y == collect(x) .+ collect(y)
        end
        @test x_zero .+ y_zero isa Zeros
        @test x_zero .+ y_one isa Ones
        @test x_one .+ y_zero isa Ones

        for x in [x, x_one, x_zero, x_dense], y in [y, y_one, y_zero, y_dense]
            @test x .* y == collect(x) .* collect(y)
        end
        for x in [x, x_one, x_zero, x_dense]
            @test x .* y_zero isa Zeros
        end
        for y in [y, y_one, y_zero, y_dense]
            @test x_zero .* y isa Zeros
        end
    end

    @test Zeros{Int}(5) .^ 0 ≡ Ones{Int}(5)
    @test Zeros{Int}(5) .^ 1 ≡ Zeros{Int}(5)
    @test Zeros{Int}(5) .+ Zeros(5) isa Zeros{Float64}

    # Test for conj, real and imag with complex element types
    @test conj(Zeros{ComplexF64}(10)) isa Zeros{ComplexF64}
    @test conj(Zeros{ComplexF64}(10,10)) isa Zeros{ComplexF64}
    @test conj(Ones{ComplexF64}(10)) isa Ones{ComplexF64}
    @test conj(Ones{ComplexF64}(10,10)) isa Ones{ComplexF64}
    @test real(Zeros{Float64}(10)) isa Zeros{Float64}
    @test real(Zeros{Float64}(10,10)) isa Zeros{Float64}
    @test real(Zeros{ComplexF64}(10)) isa Zeros{Float64}
    @test real(Zeros{ComplexF64}(10,10)) isa Zeros{Float64}
    @test real(Ones{Float64}(10)) isa Ones{Float64}
    @test real(Ones{Float64}(10,10)) isa Ones{Float64}
    @test real(Ones{ComplexF64}(10)) isa Ones{Float64}
    @test real(Ones{ComplexF64}(10,10)) isa Ones{Float64}
    @test imag(Zeros{Float64}(10)) isa Zeros{Float64}
    @test imag(Zeros{Float64}(10,10)) isa Zeros{Float64}
    @test imag(Zeros{ComplexF64}(10)) isa Zeros{Float64}
    @test imag(Zeros{ComplexF64}(10,10)) isa Zeros{Float64}
    @test imag(Ones{Float64}(10)) isa Zeros{Float64}
    @test imag(Ones{Float64}(10,10)) isa Zeros{Float64}
    @test imag(Ones{ComplexF64}(10)) isa Zeros{Float64}
    @test imag(Ones{ComplexF64}(10,10)) isa Zeros{Float64}

    @testset "range broadcast" begin
        rnge = range(-5.0, step=1.0, length=10)
        @test broadcast(*, Fill(5.0, 10), rnge) == broadcast(*, 5.0, rnge)
        @test broadcast(*, Zeros(10, 10), rnge) ≡ Zeros{Float64}(10, 10)
        @test broadcast(*, rnge, Zeros(10, 10)) ≡ Zeros{Float64}(10, 10)
        @test broadcast(*, Ones{Int}(10), rnge) ≡ rnge
        @test broadcast(*, rnge, Ones{Int}(10)) ≡ rnge
        @test broadcast(*, Ones(10), -5:4) ≡ broadcast(*, -5:4, Ones(10)) ≡ rnge
        @test broadcast(*, Ones(10), -5:1:4) ≡ broadcast(*, -5:1:4, Ones(10)) ≡ rnge
        @test_throws DimensionMismatch broadcast(*, Fill(5.0, 11), rnge)
        @test broadcast(*, rnge, Fill(5.0, 10)) == broadcast(*, rnge, 5.0)
        @test_throws DimensionMismatch broadcast(*, rnge, Fill(5.0, 11))

        # following should pass using alternative implementation in code
        deg = 5:5
        @test_throws ArgumentError @inferred(broadcast(*, Fill(5.0, 10), deg)) == broadcast(*, fill(5.0,10), deg)
        @test_throws ArgumentError @inferred(broadcast(*, deg, Fill(5.0, 10))) == broadcast(*, deg, fill(5.0,10))

        @test rnge .+ Zeros(10) ≡ rnge .- Zeros(10) ≡ Zeros(10) .+ rnge ≡ rnge

        @test_throws DimensionMismatch rnge .+ Zeros(5)
        @test_throws DimensionMismatch rnge .- Zeros(5)
        @test_throws DimensionMismatch Zeros(5) .+ rnge

        @test Fill(2,10) + (1:10) isa UnitRange
        @test (1:10) + Fill(2,10) isa UnitRange

        f = Fill(1+im,10)
        @test f + rnge isa AbstractRange
        @test f + rnge == rnge + f
        @test f + (1:10) isa AbstractRange
    end

    @testset "Special Zeros/Ones" begin
        @test broadcast(+,Zeros(5)) ≡ broadcast(-,Zeros(5)) ≡ Zeros(5)
        @test broadcast(+,Ones(5)) ≡ Ones(5)

        @test Zeros(5) .* Ones(5) ≡ Zeros(5) .* 1 ≡ Zeros(5)
        @test Zeros(5) .* Fill(5.0, 5) ≡ Zeros(5) .* 5.0 ≡ Zeros(5)
        @test Ones(5) .* Zeros(5) ≡ 1 .* Zeros(5) ≡ Zeros(5)
        @test Fill(5.0, 5) .* Zeros(5) ≡ 5.0 .* Zeros(5) ≡ Zeros(5)

        @test Zeros(5) ./ Ones(5) ≡ Zeros(5) ./ 1 ≡ Zeros(5)
        @test Zeros(5) ./ Fill(5.0, 5) ≡ Zeros(5) ./ 5.0 ≡ Zeros(5)
        @test Ones(5) .\ Zeros(5) ≡ 1 .\ Zeros(5) ≡ Zeros(5)
        @test Fill(5.0, 5) .\ Zeros(5) ≡ 5.0 .\ Zeros(5) ≡ Zeros(5)

        @test conj.(Zeros(5)) ≡ Zeros(5)
        @test conj.(Zeros{ComplexF64}(5)) ≡ Zeros{ComplexF64}(5)

        @test_throws DimensionMismatch broadcast(*, Ones(3), 1:6)
        @test_throws DimensionMismatch broadcast(*, 1:6, Ones(3))
        @test_throws DimensionMismatch broadcast(*, Fill(1,3), 1:6)
        @test_throws DimensionMismatch broadcast(*, 1:6, Fill(1,3))

        @testset "Number" begin
            @test broadcast(*, Zeros(5), 2) ≡ broadcast(*, 2, Zeros(5)) ≡ Zeros(5)
        end

        @testset "Nested" begin
            @test randn(5) .\ rand(5) .* Zeros(5) ≡ Zeros(5)
            @test broadcast(*, Zeros(5), Base.Broadcast.broadcasted(\, randn(5), rand(5))) ≡ Zeros(5)
        end

        @testset "array-valued" begin
            @test broadcast(*, Fill([1,2],3), 1:3) == broadcast(*, 1:3, Fill([1,2],3)) == broadcast(*, 1:3, fill([1,2],3))
            @test broadcast(*, Fill([1,2],3), Zeros(3)) == broadcast(*, Zeros(3), Fill([1,2],3)) == broadcast(*, zeros(3), fill([1,2],3))
            @test broadcast(*, Fill([1,2],3), Zeros(3)) isa Fill{Vector{Float64}}
            @test broadcast(*, [[1,2], [3,4,5]], Zeros(2)) == broadcast(*, Zeros(2), [[1,2], [3,4,5]]) == broadcast(*, zeros(2), [[1,2], [3,4,5]])
        end

        @testset "NaN" begin
            @test Zeros(5) ./ Zeros(5) ≡ Zeros(5) .\ Zeros(5) ≡ Fill(NaN,5)
            @test Zeros{Int}(5,6) ./ Zeros{Int}(5) ≡ Zeros{Int}(5) .\ Zeros{Int}(5,6) ≡ Fill(NaN,5,6)
        end

        @testset "Addition/Subtraction" begin
            @test Zeros{Int}(5) .+ (1:5) ≡ (1:5) .+ Zeros{Int}(5) ≡ (1:5) .- Zeros{Int}(5) ≡ 1:5
            @test Zeros{Int}(1) .+ (1:5) ≡ (1:5) .+ Zeros{Int}(1) ≡ (1:5) .- Zeros{Int}(1) ≡ 1:5
            @test Zeros(5) .+ (1:5) == (1:5) .+ Zeros(5) == (1:5) .- Zeros(5) == 1:5
            @test Zeros{Int}(5) .+ Fill(1,5) ≡ Fill(1,5) .+ Zeros{Int}(5) ≡ Fill(1,5) .- Zeros{Int}(5) ≡ Fill(1,5)
            @test_throws DimensionMismatch Zeros{Int}(2) .+ (1:5)
            @test_throws DimensionMismatch (1:5) .+ Zeros{Int}(2)

            for v in (rand(Bool, 5), [1:5;], SVector{5}(1:5), SVector{5,ComplexF16}(1:5)), T in (Bool, Int, Float64)
                TT = eltype(v + zeros(T, 5))
                S = v isa SVector ? SVector{5,TT} : Vector{TT}

                a = @inferred(Zeros{T}(5) .+ v)
                b = @inferred(v .+ Zeros{T}(5))
                c = @inferred(v .- Zeros{T}(5))
                @test a == b == c == v
                d = @inferred(Zeros{T}(5) .- v)
                @test d == -v
                @test all(Base.Fix2(isa, S), (a,b,c,d))
            end
        end
    end

    @testset "support Ref" begin
        @test Fill(1,10) .- 1 ≡ Fill(1,10) .- Ref(1) ≡ Fill(1,10) .- Ref(1I)
        @test Fill([1 2; 3 4],10) .- Ref(1I) == Fill([0 2; 3 3],10)
        @test Ref(1I) .+ Fill([1 2; 3 4],10) == Fill([2 2; 3 5],10)
    end

    @testset "Special Ones" begin
        @test Ones{Int}(5) .* (1:5) ≡ (1:5) .* Ones{Int}(5) ≡ 1:5
        @test Ones(5) .* (1:5) ≡ (1:5) .* Ones(5) ≡ 1.0:5
        @test Ones{Int}(5) .* Ones{Int}(5) ≡ Ones{Int}(5)
        @test Ones{Int}(5,2) .* (1:5) == Array(Ones{Int}(5,2)) .* Array(1:5)
        @test (1:5) .* Ones{Int}(5,2)  == Array(1:5) .* Array(Ones{Int}(5,2))
        @test (1:0.5:5) .* Ones{Int}(9,2)  == Array(1:0.5:5) .* Array(Ones{Int}(9,2))
        @test Ones{Int}(9,2) .* (1:0.5:5)  == Array(Ones{Int}(9,2)) .* Array(1:0.5:5)
        @test_throws DimensionMismatch Ones{Int}(6) .* (1:5)
        @test_throws DimensionMismatch (1:5) .* Ones{Int}(6)
        @test_throws DimensionMismatch Ones{Int}(5) .* Ones{Int}(6)
    end

    @testset "Zeros -" begin
        @test Zeros(10) - Zeros(10) ≡ Zeros(10)
        @test Ones(10) - Zeros(10) ≡ Ones(10)
        @test Ones(10) - Ones(10) ≡ Zeros(10)
        @test Fill(1,10) - Zeros(10) ≡ Fill(1.0,10)

        @test Zeros(10) .- Zeros(10) ≡ Zeros(10)
        @test Ones(10) .- Zeros(10) ≡ Ones(10)
        @test Ones(10) .- Ones(10) ≡ Zeros(10)
        @test Fill(1,10) .- Zeros(10) ≡ Fill(1.0,10)

        @test Zeros(10) .- Zeros(1,9) ≡ Zeros(10,9)
        @test Ones(10) .- Zeros(1,9) ≡ Ones(10,9)
        @test Ones(10) .- Ones(1,9) ≡ Zeros(10,9)

    end

    @testset "issue #208" begin
        TS = (Bool, Int, Float32, Float64)
        for S in TS, T in TS
            u = rand(S, 2)
            v = Zeros(T, 2)
            if zero(S) + zero(T) isa S
                @test @inferred(Broadcast.broadcasted(-, u, v)) === u
                @test @inferred(Broadcast.broadcasted(+, u, v)) === u
                @test @inferred(Broadcast.broadcasted(+, v, u)) === u
            else
                @test @inferred(Broadcast.broadcasted(-, u, v)) isa Broadcast.Broadcasted
                @test @inferred(Broadcast.broadcasted(+, u, v)) isa Broadcast. Broadcasted
                @test @inferred(Broadcast.broadcasted(+, v, u)) isa Broadcast.Broadcasted
            end
            @test @inferred(Broadcast.broadcasted(-, v, u)) isa Broadcast.Broadcasted
        end
    end

    @testset "Zero .*" begin
        TS = (Bool, Int, Float32, Float64)
        for S in TS, T in TS
            U = typeof(zero(S) * zero(T))
            @test Zeros{S}(10) .* Zeros{T}(10) ≡ Zeros{U}(10)
            @test rand(S, 10) .* Zeros(T, 10) ≡ Zeros(U, 10)
            @test Zeros(S, 10) .* rand(T, 10) ≡ Zeros(U, 10)
            if S !== Bool
                @test (S(1):S(10)) .* Zeros(T, 10) ≡ Zeros(U, 10)
                @test_throws DimensionMismatch (S(1):S(11)) .* Zeros(T, 10)
            end
            if T !== Bool
                @test Zeros(S, 10) .* (T(1):T(10)) ≡ Zeros(U, 10)
                @test_throws DimensionMismatch Zeros(S, 10) .* (T(1):T(11))
            end
        end
    end
end

@testset "map" begin
    x1 = Ones(5)
    @test map(exp,x1) === Fill(exp(1.0),5)
    @test map(isone,x1) === Fill(true,5)

    x0 = Zeros(5)
    @test map(exp,x0) === exp.(x0)

    x2 = Fill(2,5,3)
    @test map(exp,x2) === Fill(exp(2),5,3)

    @test map(+, x1, x2) === Fill(3.0, 5)
    @test map(+, x2, x2) === x2 .+ x2
    @test_throws DimensionMismatch map(+, x2', x2)

    # Issue https://github.com/JuliaArrays/FillArrays.jl/issues/179
    if VERSION < v"1.11.0-"  # In 1.11, 1-arg map & mapreduce was removed
        @test map(() -> "ok") == "ok"  # was MethodError: reducing over an empty collection is not allowed
        @test mapreduce(() -> "ok", *) == "ok"
    else
        @test_throws "no method matching map" map(() -> "ok")
        @test_throws "no method matching map" mapreduce(() -> "ok", *)
    end
end

@testset "mapreduce" begin
    x = rand(3, 4)
    y = fill(1.0, 3, 4)
    Y = Fill(1.0, 3, 4)
    O = Ones(3, 4)

    @test mapreduce(exp, +, Y) == mapreduce(exp, +, y)
    @test mapreduce(exp, +, Y; dims=2) == mapreduce(exp, +, y; dims=2)
    @test mapreduce(identity, +, Y) == sum(y) == sum(Y)
    @test mapreduce(identity, +, Y, dims=1) == sum(y, dims=1) == sum(Y, dims=1)

    @test mapreduce(exp, +, Y; dims=(1,), init=5.0) == mapreduce(exp, +, y; dims=(1,), init=5.0)

    # Two arrays
    @test mapreduce(*, +, x, Y) == mapreduce(*, +, x, y)
    @test mapreduce(*, +, Y, x) == mapreduce(*, +, y, x)
    @test mapreduce(*, +, x, O) == mapreduce(*, +, x, y)
    @test mapreduce(*, +, Y, O) == mapreduce(*, +, y, y)

    f2(x,y) = 1 + x/y
    op2(x,y) = x^2 + 3y
    @test mapreduce(f2, op2, x, Y) == mapreduce(f2, op2, x, y)

    @test mapreduce(f2, op2, x, Y, dims=1, init=5.0) == mapreduce(f2, op2, x, y, dims=1, init=5.0)
    @test mapreduce(f2, op2, Y, x, dims=1, init=5.0) == mapreduce(f2, op2, y, x, dims=1, init=5.0)
    @test mapreduce(f2, op2, x, O, dims=1, init=5.0) == mapreduce(f2, op2, x, y, dims=1, init=5.0)
    @test mapreduce(f2, op2, Y, O, dims=1, init=5.0) == mapreduce(f2, op2, y, y, dims=1, init=5.0)

    # More than two
    @test mapreduce(+, +, x, Y, x) == mapreduce(+, +, x, y, x)
    @test mapreduce(+, +, Y, x, x) == mapreduce(+, +, y, x, x)
    @test mapreduce(+, +, x, O, Y) == mapreduce(+, +, x, y, y)
    @test mapreduce(+, +, Y, O, Y) == mapreduce(+, +, y, y, y)
    @test mapreduce(+, +, Y, O, Y, x) == mapreduce(+, +, y, y, y, x)
end

@testset "Offset indexing" begin
    A = Fill(3, (Base.Slice(-1:1),))
    @test axes(A)  == (Base.Slice(-1:1),)
    @test A[0] == 3
    @test_throws BoundsError A[2]
    @test_throws BoundsError A[-2]

    A = Zeros((Base.Slice(-1:1),))
    @test axes(A)  == (Base.Slice(-1:1),)
    @test A[0] == 0
    @test_throws BoundsError A[2]
    @test_throws BoundsError A[-2]
end

@testset "0-dimensional" begin
    A = Fill{Int,0,Tuple{}}(3, ())

    @test A[] ≡ A[1] ≡ 3
    @test A ≡ Fill{Int,0}(3, ()) ≡ Fill(3, ()) ≡ Fill(3)
    @test size(A) == ()
    @test axes(A) == ()

    A = Ones{Int,0,Tuple{}}(())
    @test A[] ≡ A[1] ≡ 1
    @test A ≡ Ones{Int,0}(()) ≡ Ones{Int}(()) ≡ Ones{Int}()

    A = Zeros{Int,0,Tuple{}}(())
    @test A[] ≡ A[1] ≡ 0
    @test A ≡ Zeros{Int,0}(()) ≡ Zeros{Int}(()) ≡ Zeros{Int}()
end

@testset "unique" begin
    @test unique(Fill(12, 20)) == unique(fill(12, 20))
    @test unique(Fill(1, 0)) == []
    @test unique(Zeros(0)) isa Vector{Float64}
    @test !allunique(Fill("a", 2))
    @test allunique(Ones(0))
end

@testset "iterate" begin
    for d in (0, 1, 2, 100)
        for T in (Float64, Int)
            m = Eye(d)
            mcp = [x for x in m]
            @test mcp == m
            @test eltype(mcp) == eltype(m)
        end
    end
end

@testset "properties" begin
    for d in (0, 1, 2, 100)
        @test isone(Eye(d))
    end
end

@testset "any all iszero isone" begin
    for T in (Int, Float64, ComplexF64)
        for m in (Eye{T}(0), Eye{T}(0, 0), Eye{T}(0, 1), Eye{T}(1, 0))
            @test ! any(isone, m)
            @test ! any(iszero, m)
            @test ! all(iszero, m)
            @test ! all(isone, m)
        end
        @testset for d in (0, 1)
            for m in (Eye{T}(d), Eye{T}(d, d))
                M = Array(m)
                @test ! any(iszero, m)
                @test ! all(iszero, m)
                @test any(isone, m) == !isempty(m)
                @test all(isone, m) == !isempty(m)
                if !isempty(m)
                    @test ! any(iszero, m) == ! any(iszero, M)
                    @test ! all(iszero, m) == ! all(iszero, M)
                    @test any(isone, m) == any(isone, M)
                    @test all(isone, m) == all(isone, M)
                end
            end

            for m in (Eye{T}(d, d + 1), Eye{T}(d + 1, d))
                M = Array(m)
                @test any(iszero, m) == !isempty(m)
                @test ! all(iszero, m)
                @test any(isone, m) == !isempty(m)
                @test ! all(isone, m)
                if !isempty(M)
                    @test any(iszero, m) == any(iszero, M)
                    @test ! all(iszero, m) == ! all(iszero, M)
                    @test any(isone, m) == any(isone, M)
                    @test ! all(isone, m) == ! all(isone, M)
                end
            end

            onem = Ones{T}(d, d)
            @test isone(onem) == isone(Array(onem))
            @test iszero(onem) == isempty(onem) == iszero(Array(onem))

            if d > 0
                @test !isone(Ones{T}(d, 2d))
            end

            zerom = Zeros{T}(d, d)
            @test  isone(zerom) == isempty(zerom) == isone(Array(zerom))
            @test  iszero(zerom) == iszero(Array(zerom))

            if d > 0
                @test iszero(Zeros{T}(d, 2d))
            end

            fillm0 = Fill(T(0), d, d)
            @test   isone(fillm0) == isempty(fillm0) == isone(Array(fillm0))
            @test   iszero(fillm0) == iszero(Array(fillm0))

            fillm1 = Fill(T(1), d, d)
            @test isone(fillm1) == isone(Array(fillm1))
            @test iszero(fillm1) == isempty(fillm1) == iszero(Array(fillm1))

            fillm2 = Fill(T(2), d, d)
            @test isone(fillm2) == isempty(fillm2) == isone(Array(fillm2))
            @test iszero(fillm2) == isempty(fillm2) == iszero(Array(fillm2))
        end
        for d in (2, 3)
            for m in (Eye{T}(d), Eye{T}(d, d), Eye{T}(d, d + 2), Eye{T}(d + 2, d))
                @test any(iszero, m)
                @test ! all(iszero, m)
                @test any(isone, m)
                @test ! all(isone, m)
            end

            m1 = Ones{T}(d, d)
            @test ! isone(m1)
            @test ! iszero(m1)
            @test all(isone, m1)
            @test ! all(iszero, m1)

            m2 = Zeros{T}(d, d)
            @test ! isone(m2)
            @test iszero(m2)
            @test ! all(isone, m2)
            @test  all(iszero, m2)

            m3 = Fill(T(2), d, d)
            @test ! isone(m3)
            @test ! iszero(m3)
            @test ! all(isone, m3)
            @test ! all(iszero, m3)
            @test ! any(iszero, m3)

            m4 = Fill(T(1), d, d)
            @test ! isone(m4)
            @test ! iszero(m4)
        end
    end

    @test iszero(Zeros{SMatrix{2,2,Int,4}}(2))
    @test iszero(Fill(SMatrix{2,2}(0,0,0,0), 2))
    @test iszero(Fill(SMatrix{2,2}(0,0,0,1), 0))

    # compile-time evaluation
    @test @inferred((Z -> Val(iszero(Z)))(Zeros(3,3))) == Val(true)

    @testset "all/any" begin
        @test any(Ones{Bool}(10)) === all(Ones{Bool}(10)) === any(Fill(true,10)) === all(Fill(true,10)) === true
        @test any(Zeros{Bool}(10)) === all(Zeros{Bool}(10)) === any(Fill(false,10)) === all(Fill(false,10)) === false
        @test all(b -> ndims(b) ==  1, Fill([1,2],10))
        @test any(b -> ndims(b) ==  1, Fill([1,2],10))

        @test all(Fill(2,0))
        @test !any(Fill(2,0))
        @test any(Trues(2,0)) == any(trues(2,0))
        @test_throws TypeError all(Fill(2,2))
        @test all(iszero, Fill(missing,0)) === all(iszero, fill(missing,0)) === true
        @test all(iszero, Fill(missing,2)) === all(iszero, fill(missing,2)) === missing
        @test any(iszero, Fill(missing,0)) === any(iszero, fill(missing,0)) === false
        @test any(iszero, Fill(missing,2)) === any(iszero, fill(missing,2)) === missing
    end

    @testset "Error" begin
        @test_throws TypeError any(exp, Fill(1,5))
        @test_throws TypeError all(exp, Fill(1,5))
        @test_throws TypeError any(exp, Eye(5))
        @test_throws TypeError all(exp, Eye(5))
        @test_throws TypeError any(Fill(1,5))
        @test_throws TypeError all(Fill(1,5))
        @test_throws TypeError any(Zeros(5))
        @test_throws TypeError all(Zeros(5))
        @test_throws TypeError any(Ones(5))
        @test_throws TypeError all(Ones(5))
        @test_throws TypeError any(Eye(5))
        @test_throws TypeError all(Eye(5))
    end
end

@testset "Eye identity ops" begin
    m = Eye(10)
    D = Diagonal(Fill(2,10))

    for op in (permutedims, inv)
        @test op(m) === m
    end
    @test permutedims(D) ≡ D
    @test inv(D) ≡ Diagonal(Fill(1/2,10))

    for m in (Eye(10), Eye(10, 10), Eye(10, 8), Eye(8, 10), D)
        for op in (tril, triu, tril!, triu!)
            @test op(m) === m
        end
    end

    @test copy(m) ≡ m
    @test copy(D) ≡ D
end

@testset "Eye broadcast" begin
    E = Eye(2,3)
    M = Matrix(E)
    F = E .+ E
    @test F isa FillArrays.RectDiagonal
    @test F == M + M

    F = E .+ 1
    @test F == M .+ 1

    E = Eye((SOneTo(2), SOneTo(2)))
    @test axes(E .+ E) === axes(E)
end

@testset "Issue #31" begin
    @test convert(SparseMatrixCSC{Float64,Int64}, Zeros{Float64}(3, 3)) == spzeros(3, 3)
    @test sparse(Zeros(4, 2)) == spzeros(4, 2)
end

@testset "Adjoint/Transpose/permutedims" begin
    @test Ones{ComplexF64}(5,6)' ≡ transpose(Ones{ComplexF64}(5,6)) ≡ Ones{ComplexF64}(6,5)
    @test Zeros{ComplexF64}(5,6)' ≡ transpose(Zeros{ComplexF64}(5,6)) ≡ Zeros{ComplexF64}(6,5)
    @test Fill(1+im, 5, 6)' ≡ Fill(1-im, 6,5)
    @test transpose(Fill(1+im, 5, 6)) ≡ Fill(1+im, 6,5)
    @test Ones(5)' isa Adjoint # Vectors still need special dot product
    @test copy(Ones(5)') ≡ Ones(5)'
    @test copy(transpose(Ones(5))) ≡ transpose(Ones(5))
    @test Fill([1+im 2; 3 4; 5 6], 2,3)' == Fill([1+im 2; 3 4; 5 6]', 3,2)
    @test transpose(Fill([1+im 2; 3 4; 5 6], 2,3)) == Fill(transpose([1+im 2; 3 4; 5 6]), 3,2)

    @test permutedims(Ones(10)) ≡ Ones(1,10)
    @test permutedims(Zeros(10)) ≡ Zeros(1,10)
    @test permutedims(Fill(2.0,10)) ≡ Fill(2.0,1,10)
    @test permutedims(Ones(10,3)) ≡ Ones(3,10)
    @test permutedims(Zeros(10,3)) ≡ Zeros(3,10)
    @test permutedims(Fill(2.0,10,3)) ≡ Fill(2.0,3,10)

    @test permutedims(Ones(2,4,5), [3,2,1]) == permutedims(Array(Ones(2,4,5)), [3,2,1])
    @test permutedims(Ones(2,4,5), [3,2,1]) ≡ Ones(5,4,2)
    @test permutedims(Zeros(2,4,5), [3,2,1]) ≡ Zeros(5,4,2)
    @test permutedims(Fill(2.0,2,4,5), [3,2,1]) ≡ Fill(2.0,5,4,2)

    @testset "recursive" begin
        S = SMatrix{2,3}(1:6)
        Z = Zeros(typeof(S), 2, 3)
        Y = zeros(typeof(S), 2, 3)
        @test Z' == Y'
        @test transpose(Z) == transpose(Y)

        F = Fill(S, 2, 3)
        G = fill(S, 2, 3)
        @test F' == G'
        @test transpose(F) == transpose(G)
    end
end

@testset "reverse" begin
    for A in (Zeros{Int}(6), Ones(2,3), Fill("abc", 2, 3, 4))
        @test reverse(A) == reverse(Array(A))
        @test reverse(A, dims=1) == reverse(Array(A), dims=1)
    end
    A = Ones{Int}(6)
    @test reverse(A, 2, 4) == reverse(Array(A), 2, 4)
    @test_throws BoundsError reverse(A, 1, 10)
end

@testset "setindex!/fill!" begin
    F = Fill(1,10)
    @test (F[1] = 1) == 1
    @test_throws BoundsError (F[11] = 1)
    @test_throws ArgumentError (F[10] = 2)


    F = Fill(1,10,5)
    @test (F[1] = 1) == 1
    @test (F[3,3] = 1) == 1
    @test_throws BoundsError (F[51] = 1)
    @test_throws BoundsError (F[1,6] = 1)
    @test_throws ArgumentError (F[10] = 2)
    @test_throws ArgumentError (F[10,1] = 2)

    @test (F[:,1] .= 1) == fill(1,10)
    @test_throws ArgumentError (F[:,1] .= 2)

    @test fill!(F,1) == F
    @test_throws ArgumentError fill!(F,2)
end

@testset "mult" begin
    @test Fill(2,10)*Fill(3,1,12) == Vector(Fill(2,10))*Matrix(Fill(3,1,12))
    @test Fill(2,10)*Fill(3,1,12) ≡ Fill(6,10,12)
    @test Fill(2,3,10)*Fill(3,10,12) ≡ Fill(60,3,12)
    @test Fill(2,3,10)*Fill(3,10) ≡ Fill(60,3)
    @test_throws DimensionMismatch Fill(2,10)*Fill(3,2,12)
    @test_throws DimensionMismatch Fill(2,3,10)*Fill(3,2,12)

    f = Fill(1, (Base.IdentityUnitRange(1:3), Base.IdentityUnitRange(1:3)))
    @test f * f === Fill(size(f,2), axes(f))

    f = Fill(2, (Base.IdentityUnitRange(2:3), Base.IdentityUnitRange(2:3)))
    @test_throws ArgumentError f * f

    @test Ones(10)*Fill(3,1,12) ≡ Fill(3.0,10,12)
    @test Ones(10,3)*Fill(3,3,12) ≡ Fill(9.0,10,12)
    @test Ones(10,3)*Fill(3,3) ≡ Fill(9.0,10)

    @test Fill(2,10)*Ones(1,12) ≡ Fill(2.0,10,12)
    @test Fill(2,3,10)*Ones(10,12) ≡ Fill(20.0,3,12)
    @test Fill(2,3,10)*Ones(10) ≡ Fill(20.0,3)

    @test Ones(10)*Ones(1,12) ≡ Ones(10,12)
    @test Ones(3,10)*Ones(10,12) ≡ Fill(10.0,3,12)
    @test Ones(3,10)*Ones(10) ≡ Fill(10.0,3)

    @test Zeros(10)*Fill(3,1,12) ≡   Zeros(10,12)
    @test Zeros(10,3)*Fill(3,3,12) ≡ Zeros(10,12)
    @test Zeros(10,3)*Fill(3,3) ≡    Zeros(10)

    @test Fill(2,10)*  Zeros(1,12) ≡  Zeros(10,12)
    @test Fill(2,3,10)*Zeros(10,12) ≡ Zeros(3,12)
    @test Fill(2,3,10)*Zeros(10) ≡    Zeros(3)

    @test Zeros(10)*Zeros(1,12) ≡ Zeros(10,12)
    @test Zeros(3,10)*Zeros(10,12) ≡ Zeros(3,12)
    @test Zeros(3,10)*Zeros(10) ≡ Zeros(3)

    f = Zeros((Base.IdentityUnitRange(1:4), Base.IdentityUnitRange(1:4)))
    @test f * f === f

    f = Zeros((Base.IdentityUnitRange(3:4), Base.IdentityUnitRange(3:4)))
    @test_throws ArgumentError f * f

    @testset "Arrays as elements" begin
        SMT = SMatrix{2,2,Int,4}
        SVT = SVector{2,Int}
        @test @inferred(Zeros{SMT}(0,0) * Fill([1 2; 3 4], 0, 0)) == Zeros{SMT}(0,0)
        @test @inferred(Zeros{SMT}(4,2) * Fill([1 2; 3 4], 2, 3)) == Zeros{SMT}(4,3)
        @test @inferred(Fill([1 2; 3 4], 2, 3) * Zeros{SMT}(3, 4)) == Zeros{SMT}(2,4)
        @test @inferred(Zeros{SMT}(4,2) * Fill([1, 2], 2, 3)) == Zeros{SVT}(4,3)
        @test @inferred(Fill([1 2], 2, 3) * Zeros{SMT}(3,4)) == Zeros{SMatrix{1,2,Int,2}}(2,4)

        TSM = SMatrix{2,2,Int,4}
        s = TSM(1:4)
        for n in 0:3
            v = fill(s, 1)
            z = zeros(TSM, n)
            A = @inferred Zeros{TSM}(n) * Diagonal(v)
            B = z * Diagonal(v)
            @test A == B

            w = fill(s, n)
            A = @inferred Diagonal(w) * Zeros{TSM}(n)
            B = Diagonal(w) * z
            @test A == B

            A = @inferred Zeros{TSM}(2n, n) * Diagonal(w)
            B = zeros(TSM, 2n, n) * Diagonal(w)
            @test A == B

            A = @inferred Diagonal(w) * Zeros{TSM}(n, 2n)
            B = Diagonal(w) * zeros(TSM, n, 2n)
            @test A == B
        end

        D = Diagonal([[1 2; 3 4], [1 2 3; 4 5 6]])
        @test @inferred(Zeros(TSM, 2,2) * D) == zeros(TSM, 2,2) * D

        D = Diagonal(fill(SMatrix{2,3}(fill(im,6)),1))
        Z = Zeros(SMatrix{2,3,ComplexF64,6},1)
        @test D * Z' == fill(zero(SMatrix{2,2,ComplexF64,4}),1,1)

        D = Diagonal(fill(zeros(2,3), 2))
        Z = Zeros(SMatrix{2,3,Float64,6}, 2)
        @test Z' * D == Array(Z)' * D

        S = SMatrix{2,3}(1:6)
        A = reshape([S,2S,3S,4S],2,2)
        F = Fill(S',2,2)
        @test A * F == A * fill(S',size(F))
        @test mul!(A * F, A, F, 2, 1) == 3 * A * fill(S',size(F))
        @test F * A == fill(S',size(F)) * A
        @test mul!(F * A, F, A, 2, 1) == 3 * fill(S',size(F)) * A

        # doubly nested
        A = [[[1,2]]]'
        Z = Zeros(SMatrix{1,1,SMatrix{2,2,Int,4},1},1)
        Z2 = zeros(SMatrix{1,1,SMatrix{2,2,Int,4},1},1)
        @test A * Z == A * Z2

        x = [1 2 3; 4 5 6]
        A = reshape([x,2x,3x,4x],2,2)
        F = Fill(x,2,2)
        @test A' * F == A' * fill(x,size(F))
        @test mul!(A' * F, A', F, 2, 1) == 3 * A' * fill(x,size(F))
    end

    for W in (zeros(3,4), @MMatrix zeros(3,4))
        mW, nW = size(W)
        @test mul!(W, Fill(2,mW,5), Fill(3,5,nW)) ≈ Fill(30,mW,nW) ≈ fill(2,mW,5) * fill(3,5,nW)
        W .= 2
        @test mul!(W, Fill(2,mW,5), Fill(3,5,nW), 1.0, 2.0) ≈ Fill(30,mW,nW) .+ 4 ≈ fill(2,mW,5) * fill(3,5,nW) .+ 4
    end

    for w in (zeros(5), @MVector zeros(5))
        mw = size(w,1)
        @test mul!(w, Fill(2,mw,5), Fill(3,5)) ≈ Fill(30,mw) ≈ fill(2,mw,5) * fill(3,5)
        w .= 2
        @test mul!(w, Fill(2,mw,5), Fill(3,5), 1.0, 2.0) ≈ Fill(30,mw) .+ 4 ≈ fill(2,mw,5) * fill(3,5) .+ 4
    end

    @testset "strided" begin
        @testset for (la, (mA, nA)) in [(3, (1,4)), (0, (1,4)), (3, (1, 0))]

            a = randn(la)
            na = 1
            A = randn(mA,nA)

            @test Fill(2,3)*A ≈ Vector(Fill(2,3))*A
            @test Fill(2,0)*A ≈ Vector(Fill(2,0))*A
            @test Fill(2,3,mA)*A ≈ mul!(similar(A, 3,nA), Fill(2,3,mA), A) ≈ Matrix(Fill(2,3,mA))*A
            @test Fill(2,3,la)*a ≈ mul!(similar(a, 3), Fill(2,3,la), a) ≈ Matrix(Fill(2,3,la))*a
            @test Fill(2,3,la)*a isa Fill
            @test Ones(3)*A ≈ Vector(Ones(3))*A
            @test Ones(3,mA)*A ≈ mul!(similar(A, 3, nA), Ones(3,mA), A) ≈ Matrix(Ones(3,mA))*A
            @test Ones(3,la)*a ≈ mul!(similar(a, 3), Ones(3,la), a) ≈ Matrix(Ones(3,la))*a
            @test Ones(3,la)*a isa Fill
            @test Zeros(3)*A  ≡ Zeros(3,nA)
            @test Zeros(3,mA)*A == mul!(similar(A, 3, nA), Zeros(3,mA), A) == Zeros(3,nA)
            @test Zeros(3,la)*a == mul!(similar(A, 3), Zeros(3,la), a) == Zeros(3)

            @test A*Fill(2,nA) ≈ A*Vector(Fill(2,nA))
            @test A*Fill(2,nA,1) ≈ A*Matrix(Fill(2,nA,1))
            @test a*Fill(2,na,3) ≈ a*Matrix(Fill(2,na,3))
            @test A*Fill(2,nA,0) ≈ A*Matrix(Fill(2,nA,0))
            @test a*Fill(2,na,0) ≈ a*Matrix(Fill(2,na,0))
            @test A*Ones(nA) ≈ A*Vector(Ones(nA))
            @test A*Ones(nA,1) ≈ A*Matrix(Ones(nA,1))
            @test a*Ones(na,3) ≈ a*Matrix(Ones(na,3))
            @test A*Zeros(nA)  ≡ Zeros(mA)
            @test A*Zeros(nA,1) ≡ Zeros(mA,1)
            @test a*Zeros(na,3) ≡ Zeros(la,3)

            @test transpose(A) * Zeros(mA) ≡ Zeros(nA)
            @test A' * Zeros(mA) ≡ Zeros(nA)

            @test transpose(a) * Zeros(la, 3) ≡ transpose(Zeros(3))
            @test a' * Zeros(la,3) ≡ adjoint(Zeros(3))

            @test Zeros(la)' * Transpose(Adjoint(a)) == 0.0

            F = Fill(2, mA, 3)
            @test transpose(A) * F ≈ transpose(Fill(2, 3, mA) * A)
            F = Fill(2, la, 3)
            FS = Fill(2, (Base.OneTo(la), SOneTo(3)))
            @testset for (adjf, adjT) in ((transpose, Transpose), (adjoint, Adjoint))
                @test adjf(a) * F ≈ adjf(Fill(2, 3, la) * a)
                @test adjf(a) * F isa adjT{<:Any, <:Fill{<:Any,1}}
                @test adjf(a) * FS ≈ adjf(Fill(2, 3, la) * a)
                @test axes(adjf(a) * FS, 2) == SOneTo(3)
            end

            w = zeros(mA)
            @test mul!(w, A, Fill(2,nA), true, false) ≈ A * fill(2,nA)
            w .= 2
            @test mul!(w, A, Fill(2,nA), 1.0, 1.0) ≈ A * fill(2,nA) .+ 2

            nW = 3
            W = zeros(mA, nW)
            @test mul!(W, A, Fill(2,nA,nW), true, false) ≈ A * fill(2,nA,nW)
            W .= 2
            @test mul!(W, A, Fill(2,nA,nW), 1.0, 1.0) ≈ A * fill(2,nA,nW) .+ 2

            mW = 5
            W = zeros(mW, nA)
            @test mul!(W, Fill(2,mW,mA), A, true, false) ≈ fill(2,mW,mA) * A
            W .= 2
            @test mul!(W, Fill(2,mW,mA), A, 1.0, 1.0) ≈ fill(2,mW,mA) * A .+ 2

            mw = 5
            w = zeros(mw)
            @test mul!(w, Fill(2,mw,la), a, true, false) ≈ fill(2,mw,la) * a
            w .= 2
            @test mul!(w, Fill(2,mw,la), a, 1.0, 1.0) ≈ fill(2,mw,la) * a .+ 2

            @testset for f in [adjoint, transpose]
                w = zeros(nA)
                @test mul!(w, f(A), Fill(2,mA), true, false) ≈ f(A) * fill(2,mA)
                w .= 2
                @test mul!(w, f(A), Fill(2,mA), 1.0, 1.0) ≈ f(A) * fill(2,mA) .+ 2

                W = zeros(nA, nW)
                @test mul!(W, f(A), Fill(2,mA,nW), true, false) ≈ f(A) * fill(2,mA,nW)
                W .= 2
                @test mul!(W, f(A), Fill(2,mA,nW), 1.0, 1.0) ≈ f(A) * fill(2,mA,nW) .+ 2

                W = zeros(mW, mA)
                @test mul!(W, Fill(2,mW,nA), f(A), true, false) ≈ fill(2,mW,nA) * f(A)
                W .= 2
                @test mul!(W, Fill(2,mW,nA), f(A), 1.0, 1.0) ≈ fill(2,mW,nA) * f(A) .+ 2
            end
        end
    end

    D = Diagonal(randn(1))
    @test Zeros(1,1)*D ≡ Zeros(1,1)
    @test Zeros(1)*D ≡ Zeros(1,1)
    @test D*Zeros(1,1) ≡ Zeros(1,1)
    @test D*Zeros(1) ≡ Zeros(1)

    D = Diagonal(Fill(2,10))
    @test D * Ones(10) ≡ Fill(2.0,10)
    @test D * Ones(10,5) ≡ Fill(2.0,10,5)
    @test Ones(5,10) * D ≡ Fill(2.0,5,10)

    # following test is broken in Base as of Julia v1.5
    @test_throws DimensionMismatch Diagonal(Fill(1,1)) * Ones(10)
    @test_throws DimensionMismatch Diagonal(Fill(1,1)) * Ones(10,5)
    @test_throws DimensionMismatch Ones(5,10) * Diagonal(Fill(1,1))

    E = Eye(5)
    @test E*(1:5) ≡ 1.0:5.0
    @test (1:5)'E == (1.0:5)'
    @test E*E ≡ E

    n  = 10
    k  = 12
    m  = 15
    for T in (Float64, ComplexF64)
        Ank  = rand(T, n, k)
        Akn = rand(T, k, n)
        Ak = rand(T, k)
        onesm = ones(m)
        zerosm = zeros(m)

        fv = T == Float64 ? T(1.6) : T(1.6, 1.3)

        for (fillvec, fillmat) in ((Fill(fv, k), Fill(fv, k, m)),
                                    (Ones(T, k), Ones(T, k, m)),
                                    (Zeros(T, k), Zeros(T, k, m)))

            Afillvec = Array(fillvec)
            Afillmat = Array(fillmat)
            @test Ank * fillvec ≈ Ank * Afillvec
            @test Ank * fillmat ≈ Ank * Afillmat

            for A  in (Akn, Ak)
                @test transpose(A)*fillvec ≈ transpose(A)*Afillvec
                AtF = transpose(A)*fillmat
                AtM = transpose(A)*Afillmat
                @test AtF ≈ AtM
                @test AtF * Ones(m) ≈ AtM * onesm
                @test AtF * Zeros(m) ≈ AtM * zerosm
                @test adjoint(A)*fillvec ≈ adjoint(A)*Afillvec
                AadjF = adjoint(A)*fillmat
                AadjM = adjoint(A)*Afillmat
                @test AadjF ≈ AadjM
                @test AadjF * Ones(m) ≈ AadjM * onesm
                @test AadjF * Zeros(m) ≈ AadjM * zerosm
            end
        end

        # inplace C = F * A' * alpha + C * beta
        F = Fill(fv, m, k)
        M = Array(F)
        C = rand(T, m, n)
        @testset for f in (adjoint, transpose)
            @test mul!(copy(C), F, f(Ank)) ≈ mul!(copy(C), M, f(Ank))
            @test mul!(copy(C), F, f(Ank), 1.0, 2.0) ≈ mul!(copy(C), M, f(Ank), 1.0, 2.0)
        end
    end

    @testset "non-commutative" begin
        A = Fill(quat(rand(4)...), 2, 2)
        M = Array(A)
        α, β = quat(0,1,1,0), quat(1,0,0,1)
        @testset "matvec" begin
            f = Fill(quat(rand(4)...), size(A,2))
            v = Array(f)
            D = copy(v)
            exp_res = M * v * α + D * β
            @test mul!(copy(D), A, f, α, β) ≈ mul!(copy(D), M, v, α, β) ≈ exp_res
            @test mul!(copy(D), M, f, α, β) ≈ mul!(copy(D), M, v, α, β) ≈ exp_res
            @test mul!(copy(D), A, v, α, β) ≈ mul!(copy(D), M, v, α, β) ≈ exp_res

            @test mul!(copy(D), M', f, α, β) ≈ mul!(copy(D), M', v, α, β) ≈ M' * v * α + D * β
            @test mul!(copy(D), transpose(M), f, α, β) ≈ mul!(copy(D), transpose(M), v, α, β) ≈ transpose(M) * v * α + D * β
        end

        @testset "matmat" begin
            B = Fill(quat(rand(4)...), 2, 2)
            N = Array(B)
            D = copy(N)
            exp_res = M * N * α + D * β
            @test mul!(copy(D), A, B, α, β) ≈ mul!(copy(D), M, N, α, β) ≈ exp_res
            @test mul!(copy(D), M, B, α, β) ≈ mul!(copy(D), M, N, α, β) ≈ exp_res
            @test mul!(copy(D), A, N, α, β) ≈ mul!(copy(D), M, N, α, β) ≈ exp_res

            @test mul!(copy(D), M', B, α, β) ≈ mul!(copy(D), M', N, α, β) ≈ M' * N * α + D * β
            @test mul!(copy(D), transpose(M), B, α, β) ≈ mul!(copy(D), transpose(M), N, α, β) ≈ transpose(M) * N * α + D * β

            @test mul!(copy(D), A, N', α, β) ≈ mul!(copy(D), M, N', α, β) ≈ M * N' * α + D * β
            @test mul!(copy(D), A, transpose(N), α, β) ≈ mul!(copy(D), M, transpose(N), α, β) ≈ M * transpose(N) * α + D * β
        end
    end

    @testset "ambiguities" begin
        UT33 = UpperTriangular(ones(3,3))
        UT11 = UpperTriangular(ones(1,1))
        @test transpose(Zeros(3)) * Transpose(Adjoint([1,2,3])) == 0
        @test Zeros(3)' * Adjoint(Transpose([1,2,3])) == 0
        @test Zeros(3)' * UT33 == Zeros(3)'
        @test transpose(Zeros(3)) * UT33 == transpose(Zeros(3))
        @test UT11 * Zeros(3)' == Zeros(1,3)
        @test UT11 * transpose(Zeros(3)) == Zeros(1,3)
        @test Zeros(2,3) * UT33 == Zeros(2,3)
        @test UT33 * Zeros(3,2) == Zeros(3,2)
        @test UT33 * Zeros(3) == Zeros(3)
        @test Diagonal([1]) * transpose(Zeros(3)) == Zeros(1,3)
        @test Diagonal([1]) * Zeros(3)' == Zeros(1,3)
    end
end

@testset "count" begin
    @test count(Ones{Bool}(10)) == count(Fill(true,10)) == 10
    @test count(Zeros{Bool}(10)) == count(Fill(false,10)) == 0
    @test count(x -> 1 ≤ x < 2, Fill(1.3,10)) == 10
    @test count(x -> 1 ≤ x < 2, Fill(2.0,10)) == 0
end

@testset "norm" begin
    for a in (Zeros{Int}(5), Zeros(5,3), Zeros(2,3,3),
                Ones{Int}(5), Ones(5,3), Ones(2,3,3),
                Fill(2.3,5), Fill([2.3,4.2],5), Fill(4)),
        p in (-Inf, 0, 0.1, 1, 2, 3, Inf)
        @test norm(a,p) ≈ norm(Array(a),p)
    end
end

@testset "kron" begin
    for T in (Fill, Zeros, Ones), sz in ((2,), (2,2))
        f = T{Int}((T == Fill ? (3,sz...) : sz)...)
        g = Ones{Int}(2)
        z = Zeros{Int}(2)
        fc = collect(f)
        gc = collect(g)
        zc = collect(z)
        @test kron(f, f) == kron(fc, fc)
        @test kron(f, f) isa T{Int,length(sz)}
        @test kron(f, g) == kron(fc, gc)
        @test kron(f, g) isa AbstractFill{Int,length(sz)}
        @test kron(g, f) == kron(gc, fc)
        @test kron(g, f) isa AbstractFill{Int,length(sz)}
        @test kron(f, z) == kron(fc, zc)
        @test kron(f, z) isa AbstractFill{Int,length(sz)}
        @test kron(z, f) == kron(zc, fc)
        @test kron(z, f) isa AbstractFill{Int,length(sz)}
        @test kron(f, f .+ 0.5) == kron(fc, fc .+ 0.5)
        @test kron(f, f .+ 0.5) isa AbstractFill{Float64,length(sz)}
        @test kron(f, g .+ 0.5) isa AbstractFill{Float64,length(sz)}
    end

    for m in (Fill(2,2,2), "a"), sz in ((2,2), (2,))
        f = Fill(m, sz)
        g = fill(m, sz)
        @test kron(f, f) == kron(g, g)
    end

    @test_throws MethodError kron(Fill("a",2), Zeros(1)) # can't multiply String and Float64

    E = Eye(2)
    K = kron(E, E)
    @test K isa Diagonal
    if VERSION >= v"1.9"
        @test K isa typeof(E)
    end
    C = collect(E)
    @test K == kron(C, C)

    E = Eye(2,3)
    K = kron(E, E)
    C = collect(E)
    @test K == kron(C, C)
    @test issparse(kron(E,E))

    E = RectDiagonal(Fill(4,3), (6,3))
    C = collect(E)
    K = kron(E, E)
    @test K == kron(C, C)
    @test issparse(K)
end

@testset "dot products" begin
    n = 15
    o = Ones(1:n)
    z = Zeros(1:n)
    D = Diagonal(o)
    Z = Diagonal(z)

    Random.seed!(5)
    u = rand(n)
    v = rand(n)
    c = rand(ComplexF16, n)

    @test dot(u, D, v) == dot(u, v)
    @test dot(u, 2D, v) == 2dot(u, v)
    @test dot(u, Z, v) == 0

    @test @inferred(dot(Zeros(5), Zeros{ComplexF16}(5))) ≡ zero(ComplexF64)
    @test @inferred(dot(Zeros(5), Ones{ComplexF16}(5))) ≡ zero(ComplexF64)
    @test abs(@inferred(dot(Ones{ComplexF16}(5), Zeros(5)))) ≡ abs(@inferred(dot(randn(5), Zeros{ComplexF16}(5)))) ≡ abs(@inferred(dot(Zeros{ComplexF16}(5), randn(5)))) ≡ zero(Float64) # 0.0 !≡ -0.0
    @test @inferred(dot(c, Fill(1 + im, 15))) ≡ (@inferred(dot(Fill(1 + im, 15), c)))' ≈ @inferred(dot(c, fill(1 + im, 15)))

    @test @inferred(dot(Fill(1,5), Fill(2.0,5))) ≡ 10.0
    @test_skip dot(Fill(true,5), Fill(Int8(1),5)) isa Int8 # not working at present

    let N = 2^big(1000) # fast dot for fast sum
        @test dot(Fill(2,N),1:N) == dot(Fill(2,N),1:N) == dot(1:N,Fill(2,N)) == 2*sum(1:N)
    end

    @test_throws DimensionMismatch dot(u[1:end-1], D, v)
    @test_throws DimensionMismatch dot(u[1:end-1], D, v[1:end-1])

    @test_throws DimensionMismatch dot(u, 2D, v[1:end-1])
    @test_throws DimensionMismatch dot(u, 2D, v[1:end-1])

    @test_throws DimensionMismatch dot(u, Z, v[1:end-1])
    @test_throws DimensionMismatch dot(u, Z, v[1:end-1])

    @test_throws DimensionMismatch dot(Zeros(5), Zeros(6))
    @test_throws DimensionMismatch dot(Zeros(5), randn(6))
end

@testset "print" begin
    # 3-arg show, full printing
    @test stringmime("text/plain", Zeros(3)) == "3-element Zeros{Float64}"
    @test stringmime("text/plain", Ones(3)) == "3-element Ones{Float64}"
    @test stringmime("text/plain", Fill(7,2)) == "2-element Fill{$Int}, with entries equal to 7"
    @test stringmime("text/plain", Zeros(3,2)) == "3×2 Zeros{Float64}"
    @test stringmime("text/plain", Ones(3,2)) == "3×2 Ones{Float64}"
    @test stringmime("text/plain", Fill(7,2,3)) == "2×3 Fill{$Int}, with entries equal to 7"
    @test stringmime("text/plain", Fill(8.0,1)) == "1-element Fill{Float64}, with entry equal to 8.0"
    @test stringmime("text/plain", Eye(5)) == "5×5 Eye{Float64}"
    # used downstream in LazyArrays.jl to deduce sparsity
    @test Base.replace_in_print_matrix(Zeros(5,3), 1, 2, "0.0") == " ⋅ "

    # 2-arg show, compact printing
    @test repr(Zeros{Int}()) == "Zeros{$Int}()"
    @test repr(Zeros{Int}(3)) == "Zeros{$Int}(3)"
    @test repr(Zeros(3)) == "Zeros(3)"
    @test repr(Ones{Int}(3)) == "Ones{$Int}(3)"
    @test repr(Ones{Int}(3,2)) == "Ones{$Int}(3, 2)"
    @test repr(Ones(3,2)) == "Ones(3, 2)"
    @test repr(Fill(7,3,2)) == "Fill(7, 3, 2)"
    @test repr(Fill(1f0,10)) == "Fill(1.0f0, 10)"  # Float32!
    @test repr(Fill(0)) == "Fill(0)"
    @test repr(Eye(9)) == "Eye(9)"
    @test repr(Eye(9,4)) == "Eye(9,4)"
    # also used for arrays of arrays:
    @test occursin("Eye(2) ", stringmime("text/plain", [Eye(2) for i in 1:2, j in 1:2]))
end

@testset "reshape" begin
    @test reshape(Fill(2,6),2,3) ≡ reshape(Fill(2,6),(2,3)) ≡ reshape(Fill(2,6),:,3) ≡ reshape(Fill(2,6),2,:) ≡ Fill(2,2,3)
    @test reshape(Fill(2,6),big(2),3) == reshape(Fill(2,6), (big(2),3)) == reshape(Fill(2,6), big(2),:) == Fill(2,big(2),3)
    @test_throws DimensionMismatch reshape(Fill(2,6),2,4)
    @test reshape(Ones(6),2,3) ≡ reshape(Ones(6),(2,3)) ≡ reshape(Ones(6),:,3) ≡ reshape(Ones(6),2,:) ≡ Ones(2,3)
    @test reshape(Zeros(6),2,3) ≡ Zeros(2,3)
    @test reshape(Zeros(6),big(2),3) == Zeros(big(2),3)
    @test reshape(Fill(2,2,3),Val(1)) ≡ Fill(2,6)
    @test reshape(Fill(2, 2), (2, )) ≡ Fill(2, 2)

    @test reshape(Fill(2,3), :) === reshape(Fill(2,3), (:,)) === Fill(2,3)
end

@testset "lmul!/rmul!" begin
    z = Zeros(1_000_000_000_000)
    @test lmul!(2.0,z) === z
    @test rmul!(z,2.0) === z
    @test_throws ArgumentError lmul!(Inf,z)
    @test_throws ArgumentError rmul!(z,Inf)

    x = Fill([1,2],1_000_000_000_000)
    @test lmul!(1.0,x) === x
    @test rmul!(x,1.0) === x
    @test_throws ArgumentError lmul!(2.0,x)
    @test_throws ArgumentError rmul!(x,2.0)
end

@testset "Modified" begin
    @testset "Diagonal{<:Fill}" begin
        D = Diagonal(Fill(Fill(0.5,2,2),10))
        @test @inferred(D[1,1]) === Fill(0.5,2,2)
        @test @inferred(D[1,2]) === Fill(0.0,2,2)
        @test axes(D) == (Base.OneTo(10),Base.OneTo(10))
        D = Diagonal(Fill(Zeros(2,2),10))
        @test @inferred(D[1,1]) === Zeros(2,2)
        @test @inferred(D[1,2]) === Zeros(2,2)
        D = Diagonal([Zeros(1,1), Zeros(2,2)])
        @test @inferred(D[1,1]) === Zeros(1,1)
        @test @inferred(D[1,2]) === Zeros(1,2)

        @test_throws ArgumentError Diagonal(Fill(Ones(2,2),10))[1,2]
    end
    @testset "Triangular" begin
        U = UpperTriangular(Ones(3,3))
        @test U == UpperTriangular(ones(3,3))
        @test axes(U) == (Base.OneTo(3),Base.OneTo(3))
    end
end

@testset "Trues" begin
    @test Trues(2,3) == Trues((2,3)) == trues(2,3)
    @test Falses(2,3) == Falses((2,3)) == falses(2,3)
    dim = (4,5)
    mask = Trues(dim)
    x = randn(dim)
    @test x[mask] == vec(x) # getindex
    y = similar(x)
    y[mask] = x # setindex!
    @test y == x
    @test_throws BoundsError ones(3)[Trues(2)]
    @test_throws BoundsError setindex!(ones(3), zeros(3), Trues(2))
    @test_throws DimensionMismatch setindex!(ones(2), zeros(3), Trues(2))
    @test Ones(3)[Trues(3)] == Ones(3)
    @test_throws BoundsError Ones(3)[Trues(2)]
    @test Ones(2,3)[Trues(2,3)] == Ones(6)
    @test Ones(2,3)[Trues(6)] == Ones(6)
    @test_throws BoundsError Ones(2,3)[Trues(3,2)]
end

@testset "FillArray interface" begin
    @testset "SubArray" begin
        a = Fill(2.0,5)
        v = SubArray(a,(1:2,))
        @test FillArrays.getindex_value(v) == FillArrays.unique_value(v) == 2.0
        @test convert(Fill, v) ≡ Fill(2.0,2)
    end

    @testset "views" begin
        a = Fill(2.0,5)
        v = view(a,1:2)
        @test v isa Fill
        @test FillArrays.getindex_value(v) == FillArrays.unique_value(v) == 2.0
        @test convert(Fill, v) ≡ Fill(2.0,2)
        @test view(a,1) ≡ Fill(2.0)
        @test view(a,1,1) ≡ Fill(2.0)
        @test view(a, :) === a
        @test view(a, CartesianIndices(a)) === a
        vv = view(a, CartesianIndices(a), :, 1)
        @test ndims(vv) == 2
        @test vv isa Fill && FillArrays.getindex_value(vv) == 2.0
        vv = view(a, CartesianIndices(a), :, 1:1)
        @test ndims(vv) == 3
        @test vv isa Fill && FillArrays.getindex_value(vv) == 2.0
    end

    @testset "view with bool" begin
        a = Fill(2.0,5)
        @test a[[true,false,false,true,false]] ≡ view(a,[true,false,false,true,false])
        a = Fill(2.0,2,2)
        @test a[[true false; false true]] ≡ view(a, [true false; false true])
    end

    @testset "adjtrans" begin
        a = Fill(2.0+im, 5)
        @test FillArrays.getindex_value(a') == FillArrays.unique_value(a') == 2.0 - im
        @test convert(Fill, a') ≡ Fill(2.0-im,1,5)
        @test FillArrays.getindex_value(transpose(a)) == FillArrays.unique_value(transpose(a)) == 2.0 + im
        @test convert(Fill, transpose(a)) ≡ Fill(2.0+im,1,5)
    end

    @testset "custom AbstractFill types" begin
        # implicit axes
        struct StaticZerosVec{L,T} <: FillArrays.AbstractZeros{T,1,Tuple{SOneTo{L}}} end
        Base.size(::StaticZerosVec{L}) where {L} = (L,)
        Base.axes(::StaticZerosVec{L}) where {L} = (SOneTo(L),)
        S = StaticZerosVec{3,Int}()
        @test real.(S) == S
        @test imag.(S) == S

        struct StaticOnesVec{L,T} <: FillArrays.AbstractOnes{T,1,Tuple{SOneTo{L}}} end
        Base.size(::StaticOnesVec{L}) where {L} = (L,)
        Base.axes(::StaticOnesVec{L}) where {L} = (SOneTo(L),)
        S = StaticOnesVec{3,Int}()
        @test real.(S) == S
        @test imag.(S) == zero(S)

        struct StaticFill{S1,S2,T} <: FillArrays.AbstractFill{T,2,Tuple{SOneTo{S1},SOneTo{S2}}}
            x :: T
        end
        StaticFill{S1,S2}(x::T) where {S1,S2,T} = StaticFill{S1,S2,T}(x)
        Base.size(::StaticFill{S1,S2}) where {S1,S2} = (S1,S2)
        Base.axes(::StaticFill{S1,S2}) where {S1,S2} = (SOneTo(S1), SOneTo(S2))
        FillArrays.getindex_value(S::StaticFill) = S.x
        S = StaticFill{2,3}(2)
        @test permutedims(S) == Fill(2, reverse(size(S)))
    end
end

@testset "Statistics" begin
    @test mean(Fill(3,4,5)) === mean(fill(3,4,5))
    @test std(Fill(3,4,5)) === std(fill(3,4,5))
    @test var(Trues(5)) === var(trues(5))
    @test mean(Trues(5)) === mean(trues(5))

    @test mean(sqrt, Fill(3,4,5)) ≈ mean(sqrt, fill(3,4,5))

    @test mean(Fill(3,4,5), dims=2) == mean(fill(3,4,5), dims=2)
    @test std(Fill(3,4,5), corrected=true, mean=3) == std(fill(3,4,5), corrected=true, mean=3)

    @test cov(Fill(3,4)) === cov(fill(3,4))
    @test cov(Fill(3,4,5)) == cov(fill(3,4,5))
    @test cov(Fill(3,4,5), dims=2) == cov(fill(3,4,5), dims=2)

    @test cor(Fill(3,4)) == cor(fill(3,4))
    @test cor(Fill(3, 4, 5)) ≈ cor(fill(3, 4, 5)) nans=true
    @test cor(Fill(3, 4, 5), dims=2) ≈ cor(fill(3, 4, 5), dims=2) nans=true
end

@testset "Structured broadcast" begin
    D = Diagonal(1:5)
    @test D + Zeros(5,5) isa Diagonal
    @test D - Zeros(5,5) isa Diagonal
    @test D .+ Zeros(5,5) isa Diagonal
    @test D .- Zeros(5,5) isa Diagonal
    @test D .* Zeros(5,5) isa Diagonal
    @test Zeros(5,5) .* D isa Diagonal
    @test Zeros(5,5) - D isa Diagonal
    @test Zeros(5,5) + D isa Diagonal
    @test Zeros(5,5) .- D isa Diagonal
    @test Zeros(5,5) .+ D isa Diagonal
    f = (x,y) -> x+1
    @test f.(D, Zeros(5,5)) isa Matrix
end

@testset "OneElement" begin
    A = OneElement(2, (), ())
    @test FillArrays.nzind(A) == CartesianIndex()
    @test A == Fill(2, ())
    @test A[] === 2
    @test A[1] === A[1,1] === 2

    e₁ = OneElement(2, 5)
    @test e₁ == [0,1,0,0,0]
    @test FillArrays.nzind(e₁) == CartesianIndex(2)
    @test e₁[2] === e₁[2,1] === e₁[2,1,1] === 1
    @test_throws BoundsError e₁[6]

    f₁ = AbstractArray{Float64}(e₁)
    @test f₁ isa OneElement{Float64,1}
    @test f₁ == e₁
    fv₁ = AbstractVector{Float64}(e₁)
    @test fv₁ isa OneElement{Float64,1}
    @test fv₁ == e₁

    @test stringmime("text/plain", e₁) == "5-element OneElement{$Int, 1, Tuple{$Int}, Tuple{Base.OneTo{$Int}}}:\n ⋅\n 1\n ⋅\n ⋅\n ⋅"

    e₁ = OneElement{Float64}(2, 5)
    @test e₁ == [0,1,0,0,0]

    v = OneElement{Float64}(2, 3, 4)
    @test v == [0,0,2,0]

    V = OneElement(2, (2,3), (3,4))
    @test V == [0 0 0 0; 0 0 2 0; 0 0 0 0]
    @test FillArrays.nzind(V) == CartesianIndex(2,3)

    Vf = AbstractArray{Float64}(V)
    @test Vf isa OneElement{Float64,2}
    @test Vf == V
    VMf = AbstractMatrix{Float64}(V)
    @test VMf isa OneElement{Float64,2}
    @test VMf == V

    @test stringmime("text/plain", V) == "3×4 OneElement{$Int, 2, Tuple{$Int, $Int}, Tuple{Base.OneTo{$Int}, Base.OneTo{$Int}}}:\n ⋅  ⋅  ⋅  ⋅\n ⋅  ⋅  2  ⋅\n ⋅  ⋅  ⋅  ⋅"

    @test Base.setindex(Zeros(5), 2, 2) ≡ OneElement(2.0, 2, 5)
    @test Base.setindex(Zeros(5,3), 2, 2, 3) ≡ OneElement(2.0, (2,3), (5,3))
    @test_throws BoundsError Base.setindex(Zeros(5), 2, 6)

    @testset "non-numeric" begin
        S = SMatrix{2,2}(1:4)
        A = OneElement(S, (2,2), (2,2))
        @test A[2,2] === S
        @test A[1,1] === A[1,2] === A[2,1] === zero(S)
    end

    @testset "Vector indexing" begin
        @testset "1D" begin
            A = OneElement(2, 2, 4)
            @test @inferred(A[:]) === @inferred(A[axes(A)...]) === A
            @test @inferred(A[3:4]) isa OneElement{Int,1}
            @test @inferred(A[3:4]) == Zeros(2)
            @test @inferred(A[1:2]) === OneElement(2, 2, 2)
            @test @inferred(A[2:3]) === OneElement(2, 1, 2)
            @test @inferred(A[Base.IdentityUnitRange(2:3)]) isa OneElement{Int,1}
            @test @inferred(A[Base.IdentityUnitRange(2:3)]) == OneElement(2,(2,),(Base.IdentityUnitRange(2:3),))
            @test A[:,:] == reshape(A, size(A)..., 1)

            @test A[reverse(axes(A,1))] == A[collect(reverse(axes(A,1)))]

            @testset "repeated indices" begin
                @test A[StepRangeLen(2, 0, 3)] == A[fill(2, 3)]
            end

            B = OneElement(2, (2,), (Base.IdentityUnitRange(-1:4),))
            @test @inferred(A[:]) === @inferred(A[axes(A)...]) === A
            @test @inferred(A[3:4]) isa OneElement{Int,1}
            @test @inferred(A[3:4]) == Zeros(2)
            @test @inferred(A[2:3]) === OneElement(2, 1, 2)

            C = OneElement(2, (2,), (Base.OneTo(big(4)),))
            @test @inferred(C[1:4]) === OneElement(2, 2, 4)

            D = OneElement(2, (2,), (InfiniteArrays.OneToInf(),))
            D2 = D[:]
            @test axes(D2) == axes(D)
            @test D2[2] == D[2]
            D3 = D[axes(D)...]
            @test axes(D3) == axes(D)
            @test D3[2] == D[2]
        end
        @testset "2D" begin
            A = OneElement(2, (2,3), (4,5))
            @test @inferred(A[:,:]) === @inferred(A[axes(A)...]) === A
            @test @inferred(A[:,1]) isa OneElement{Int,1}
            @test @inferred(A[:,1]) == Zeros(4)
            @test A[:, Int64(1)] === A[:, Int32(1)]
            @test @inferred(A[1,:]) isa OneElement{Int,1}
            @test @inferred(A[1,:]) == Zeros(5)
            @test @inferred(A[:,3]) === OneElement(2, 2, 4)
            @test @inferred(A[2,:]) === OneElement(2, 3, 5)
            @test @inferred(A[1:1,:]) isa OneElement{Int,2}
            @test @inferred(A[1:1,:]) == Zeros(1,5)
            @test @inferred(A[4:4,:]) isa OneElement{Int,2}
            @test @inferred(A[4:4,:]) == Zeros(1,5)
            @test @inferred(A[2:2,:]) === OneElement(2, (1,3), (1,5))
            @test @inferred(A[1:4,:]) === OneElement(2, (2,3), (4,5))
            @test @inferred(A[:,3:3]) === OneElement(2, (2,1), (4,1))
            @test @inferred(A[:,1:5]) === OneElement(2, (2,3), (4,5))
            @test @inferred(A[1:4,1:4]) === OneElement(2, (2,3), (4,4))
            @test @inferred(A[2:4,2:4]) === OneElement(2, (1,2), (3,3))
            @test @inferred(A[2:4,3:4]) === OneElement(2, (1,1), (3,2))
            @test @inferred(A[4:4,5:5]) isa OneElement{Int,2}
            @test @inferred(A[4:4,5:5]) == Zeros(1,1)
            @test @inferred(A[Base.IdentityUnitRange(2:4), :]) isa OneElement{Int,2}
            @test axes(A[Base.IdentityUnitRange(2:4), :]) == (Base.IdentityUnitRange(2:4), axes(A,2))
            @test @inferred(A[:,:,:]) == reshape(A, size(A)...,1)

            B = OneElement(2, (2,3), (Base.IdentityUnitRange(2:4),Base.IdentityUnitRange(2:5)))
            @test @inferred(B[:,:]) === @inferred(B[axes(B)...])  === B
            @test @inferred(B[:,3]) === OneElement(2, (2,), (Base.IdentityUnitRange(2:4),))
            @test @inferred(B[3:4, 4:5]) isa OneElement{Int,2}
            @test @inferred(B[3:4, 4:5]) == Zeros(2,2)
            b = @inferred(B[Base.IdentityUnitRange(3:4), Base.IdentityUnitRange(4:5)])
            @test b == Zeros(axes(b))

            C = OneElement(2, (2,3), (Base.OneTo(big(4)), Base.OneTo(big(5))))
            @test @inferred(C[1:4, 1:5]) === OneElement(2, (2,3), Int.(size(C)))

            D = OneElement(2, (2,3), (InfiniteArrays.OneToInf(), InfiniteArrays.OneToInf()))
            D2 = @inferred D[:,:]
            @test axes(D2) == axes(D)
            @test D2[2,3] == D[2,3]
            D3 = @inferred D[axes(D)...]
            @test axes(D3) == axes(D)
            @test D3[2,3] == D[2,3]
        end
    end

    @testset "adjoint/transpose" begin
        A = OneElement(3im, (2,4), (4,6))
        @test A' === OneElement(-3im, (4,2), (6,4))
        @test transpose(A) === OneElement(3im, (4,2), (6,4))

        A = OneElement(3im, 2, 3)
        @test A' isa Adjoint
        @test transpose(A) isa Transpose
        @test A' == OneElement(-3im, (1,2), (1,3))
        @test transpose(A) == OneElement(3im, (1,2), (1,3))

        A = OneElement(3, (2,2), (4,4))
        @test adjoint(A) === A
        @test transpose(A) === A

        A = OneElement(3, 2, 4)
        @test transpose(A) isa Transpose
        @test adjoint(A) isa Adjoint
        @test transpose(A) == OneElement(3, (1,2), (1,4))
        @test adjoint(A) == OneElement(3, (1,2), (1,4))
    end

    @testset "reshape" begin
        for O in (OneElement(2, (2,3), (4,5)), OneElement(2, (2,), (20,)),
                    OneElement(2, (1,2,2), (2,2,5)))
            A = Array(O)
            for shp in ((2,5,2), (5,1,4), (20,), (2,2,5,1,1))
                @test reshape(O, shp) == reshape(A, shp)
            end
        end
        O = OneElement(2, (), ())
        @test reshape(O, ()) === O
    end

    @testset "isassigned" begin
        f = OneElement(2, (3,3), (4,4))
        @test !isassigned(f, 0, 0)
        @test isassigned(f, 2, 2)
        @test !isassigned(f, 10, 10)
        @test_throws ArgumentError isassigned(f, true)
    end

    @testset "matmul" begin
        A = reshape(Float64[1:9;], 3, 3)
        v = reshape(Float64[1:3;], 3)
        testinds(w::AbstractArray) = testinds(size(w))
        testinds(szw::Tuple{Int}) = (szw .- 1, szw .+ 1)
        function testinds(szA::Tuple{Int,Int})
            (szA .- 1, szA .+ (-1,0), szA .+ (0,-1), szA .+ 1, szA .+ (1,-1), szA .+ (-1,1))
        end
        # test matvec if w is a vector, or matmat if w is a matrix
        function test_mat_mul_OneElement(A, (w, w2), sz)
            @testset for ind in testinds(sz)
                x = OneElement(3, ind, sz)
                xarr = Array(x)
                Axarr = A * xarr
                Aadjxarr = A' * xarr

                @test A * x ≈ Axarr
                @test A' * x ≈ Aadjxarr
                @test transpose(A) * x ≈ transpose(A) * xarr

                @test mul!(w, A, x) ≈ Axarr
                # check columnwise to ensure zero columns
                @test all(((c1, c2),) -> c1 ≈ c2, zip(eachcol(w), eachcol(Axarr)))
                @test mul!(w, A', x) ≈ Aadjxarr
                w .= 1
                @test mul!(w, A, x, 1.0, 2.0) ≈ Axarr .+ 2
                w .= 1
                @test mul!(w, A', x, 1.0, 2.0) ≈ Aadjxarr .+ 2

                F = Fill(3, size(A))
                w2 .= 1
                @test mul!(w2, F, x, 1.0, 1.0) ≈ Array(F) * xarr .+ 1
            end
        end
        function test_OneElementMatrix_mul_mat(A, (w, w2), sz)
            @testset for ind in testinds(sz)
                O = OneElement(3, ind, sz)
                Oarr = Array(O)
                OarrA = Oarr * A
                OarrAadj = Oarr * A'

                @test O * A ≈ OarrA
                @test O * A' ≈ OarrAadj
                @test O * transpose(A) ≈ Oarr * transpose(A)

                @test mul!(w, O, A) ≈ OarrA
                # check columnwise to ensure zero columns
                @test all(((c1, c2),) -> c1 ≈ c2, zip(eachcol(w), eachcol(OarrA)))
                @test mul!(w, O, A') ≈ OarrAadj
                w .= 1
                @test mul!(w, O, A, 1.0, 2.0) ≈ OarrA .+ 2
                w .= 1
                @test mul!(w, O, A', 1.0, 2.0) ≈ OarrAadj .+ 2

                F = Fill(3, size(A))
                w2 .= 1
                @test mul!(w2, O, F, 1.0, 1.0) ≈ Oarr * Array(F) .+ 1
            end
        end
        function test_OneElementMatrix_mul_vec(v, (w, w2), sz)
            @testset for ind in testinds(sz)
                O = OneElement(3, ind, sz)
                Oarr = Array(O)
                Oarrv = Oarr * v

                @test O * v == Oarrv

                @test mul!(w, O, v) == Oarrv
                # check rowwise to ensure zero rows
                @test all(((r1, r2),) -> r1 == r2, zip(eachrow(w), eachrow(Oarrv)))
                w .= 1
                @test mul!(w, O, v, 1.0, 2.0) == Oarrv .+ 2

                F = Fill(3, size(v))
                w2 .= 1
                @test mul!(w2, O, F, 1.0, 1.0) == Oarr * Array(F) .+ 1
            end
        end
        @testset "Matrix * OneElementVector" begin
            w = zeros(size(A,1))
            w2 = MVector{length(w)}(w)
            test_mat_mul_OneElement(A, (w, w2), size(w))
        end
        @testset "Matrix * OneElementMatrix" begin
            C = zeros(size(A))
            C2 = MMatrix{size(C)...}(C)
            test_mat_mul_OneElement(A, (C, C2), size(C))
        end
        @testset "OneElementMatrix * Vector" begin
            w = zeros(size(v))
            w2 = MVector{size(v)...}(v)
            test_OneElementMatrix_mul_vec(v, (w, w2), size(A))
        end
        @testset "OneElementMatrix * Matrix" begin
            C = zeros(size(A))
            C2 = MMatrix{size(C)...}(C)
            test_OneElementMatrix_mul_mat(A, (C, C2), size(A))
        end
        @testset "OneElementMatrix * OneElement" begin
            @testset for ind in testinds(A)
                O = OneElement(3, ind, size(A))
                v = OneElement(4, ind[2], size(A,1))
                @test O * v isa OneElement
                @test O * v == Array(O) * Array(v)
                @test mul!(ones(size(O,1)), O, v) == O * v
                @test mul!(ones(size(O,1)), O, v, 2, 1) == 2 * O * v .+ 1

                B = OneElement(4, ind, size(A))
                @test O * B isa OneElement
                @test O * B == Array(O) * Array(B)
                @test mul!(ones(size(O,1), size(B,2)), O, B) == O * B
                @test mul!(ones(size(O,1), size(B,2)), O, B, 2, 1) == 2 * O * B .+ 1
            end

            @test OneElement(3, (2,3), (5,4)) * OneElement(2, 2, 4) == Zeros(5)
            @test OneElement(3, (2,3), (5,4)) * OneElement(2, (2,1), (4,2)) == Zeros(5,2)
        end
        @testset "AbstractFillMatrix * OneElementVector" begin
            F = Fill(3, size(A))
            sw = (size(A,1),)
            @testset for ind in testinds(sw)
                x = OneElement(3, ind, sw)
                @test F * x isa Fill
                @test F * x == Array(F) * Array(x)
            end

            @test Zeros{Int8}(2,2) * OneElement{Int16}(2,2) === Zeros{Int16}(2)
        end
        @testset "OneElementMatrix * AbstractFillVector" begin
            @testset for ind in testinds(A)
                O = OneElement(3, ind, size(A))
                v = Fill(2, size(O,1))
                @test O * v isa OneElement
                @test O * v == Array(O) * Array(v)
            end

            A = OneElement(2,(2,2),(5,4))
            B = Zeros(4)
            @test A * B === Zeros(5)
        end
        @testset "Diagonal and OneElementMatrix" begin
            for ind in ((2,3), (2,2), (10,10))
                O = OneElement(3, ind, (4,3))
                Oarr = Array(O)
                C = zeros(size(O))
                D = Diagonal(axes(O,1))
                @test D * O == D * Oarr
                @test mul!(C, D, O) == D * O
                C .= 1
                @test mul!(C, D, O, 2, 2) == 2 * D * O .+ 2
                D = Diagonal(axes(O,2))
                @test O * D == Oarr * D
                @test mul!(C, O, D) == O * D
                C .= 1
                @test mul!(C, O, D, 2, 2) == 2 * O * D .+ 2
            end
        end
        @testset "array elements" begin
            A = [SMatrix{2,3}(1:6)*(i+j) for i in 1:3, j in 1:2]
            @testset "StridedMatrix * OneElementMatrix" begin
                B = OneElement(SMatrix{3,2}(1:6), (size(A,2),2), (size(A,2),4))
                C = [SMatrix{2,2}(1:4) for i in axes(A,1), j in axes(B,2)]
                @test mul!(copy(C), A, B) == A * B
                @test mul!(copy(C), A, B, 2, 2) == 2 * A * B + 2 * C
            end
            @testset "StridedMatrix * OneElementVector" begin
                B = OneElement(SMatrix{3,2}(1:6), (size(A,2),), (size(A,2),))
                C = [SMatrix{2,2}(1:4) for i in axes(A,1)]
                @test mul!(copy(C), A, B) == A * B
                @test mul!(copy(C), A, B, 2, 2) == 2 * A * B + 2 * C
            end

            A = OneElement(SMatrix{3,2}(1:6), (3,2), (5,4))
            @testset "OneElementMatrix * StridedMatrix" begin
                B = [SMatrix{2,3}(1:6)*(i+j) for i in axes(A,2), j in 1:2]
                C = [SMatrix{3,3}(1:9) for i in axes(A,1), j in axes(B,2)]
                @test mul!(copy(C), A, B) == A * B
                @test mul!(copy(C), A, B, 2, 2) == 2 * A * B + 2 * C
            end
            @testset "OneElementMatrix * StridedVector" begin
                B = [SMatrix{2,3}(1:6)*i for i in axes(A,2)]
                C = [SMatrix{3,3}(1:9) for i in axes(A,1)]
                @test mul!(copy(C), A, B) == A * B
                @test mul!(copy(C), A, B, 2, 2) == 2 * A * B + 2 * C
            end
            @testset "OneElementMatrix * OneElementMatrix" begin
                B = OneElement(SMatrix{2,3}(1:6), (2,4), (size(A,2), 3))
                C = [SMatrix{3,3}(1:9) for i in axes(A,1), j in axes(B,2)]
                @test mul!(copy(C), A, B) == A * B
                @test mul!(copy(C), A, B, 2, 2) == 2 * A * B + 2 * C
            end
            @testset "OneElementMatrix * OneElementVector" begin
                B = OneElement(SMatrix{2,3}(1:6), 2, size(A,2))
                C = [SMatrix{3,3}(1:9) for i in axes(A,1)]
                @test mul!(copy(C), A, B) == A * B
                @test mul!(copy(C), A, B, 2, 2) == 2 * A * B + 2 * C
            end
        end
        @testset "non-commutative" begin
            A = OneElement(quat(rand(4)...), (2,3), (3,4))
            for (B,C) in (
                        # OneElementMatrix * OneElementVector
                        (OneElement(quat(rand(4)...), 3, size(A,2)),
                            [quat(rand(4)...) for i in axes(A,1)]),

                        # OneElementMatrix * OneElementMatrix
                        (OneElement(quat(rand(4)...), (3,2), (size(A,2), 4)),
                            [quat(rand(4)...) for i in axes(A,1), j in 1:4]),
                        )
                @test mul!(copy(C), A, B) ≈ A * B
                α, β = quat(0,0,1,0), quat(1,0,1,0)
                @test mul!(copy(C), A, B, α, β) ≈ mul!(copy(C), A, Array(B), α, β) ≈ A * B * α + C * β
            end

            A = [quat(rand(4)...)*(i+j) for i in 1:2, j in 1:3]
            for (B,C) in (
                        # StridedMatrix * OneElementVector
                        (OneElement(quat(rand(4)...), 1, size(A,2)),
                            [quat(rand(4)...) for i in axes(A,1)]),

                        # StridedMatrix * OneElementMatrix
                        (OneElement(quat(rand(4)...), (2,2), (size(A,2), 4)),
                            [quat(rand(4)...) for i in axes(A,1), j in 1:4]),
                        )
                @test mul!(copy(C), A, B) ≈ A * B
                α, β = quat(0,0,1,0), quat(1,0,1,0)
                @test mul!(copy(C), A, B, α, β) ≈ mul!(copy(C), A, Array(B), α, β) ≈ A * B * α + C * β
            end

            A = OneElement(quat(rand(4)...), (2,2), (3, 4))
            for (B,C) in (
                        # OneElementMatrix * StridedMatrix
                        ([quat(rand(4)...) for i in axes(A,2), j in 1:3],
                            [quat(rand(4)...) for i in axes(A,1), j in 1:3]),

                        # OneElementMatrix * StridedVector
                        ([quat(rand(4)...) for i in axes(A,2)],
                            [quat(rand(4)...) for i in axes(A,1)]),
                        )
                @test mul!(copy(C), A, B) ≈ A * B
                α, β = quat(0,0,1,0), quat(1,0,1,0)
                @test mul!(copy(C), A, B, α, β) ≈ mul!(copy(C), A, Array(B), α, β) ≈ A * B * α + C * β
            end
        end
    end

    @testset "multiplication/division by a number" begin
        val = 2
        x = OneElement(val,1,5)
        y = sparse(x)
        @test 3x === OneElement(3val,1,5) == 3y
        @test 3.0 * x === OneElement(3.0*val,1,5) == 3.0 * y
        @test 3.0 \ x === OneElement(3.0 \ val,1,5) == 3.0 \ y
        @test x * 3.0 === OneElement(val * 3.0,1,5) == y * 3.0
        @test x / 3.0 === OneElement(val / 3.0,1,5) == y / 3.0

        val = 3im
        A = OneElement(val, (2,2), (3,4))
        B = sparse(A)
        @test 3A === OneElement(3val, (2,2), (3,4)) == 3B
        @test 3.0im * A === OneElement(3.0im * val, (2,2), (3,4)) == 3.0im * B
        @test 3.0im \ A === OneElement(3.0im \ val, (2,2), (3,4)) == 3.0im \ B
        @test A * (2 + 3.0im) === OneElement(val * (2 + 3.0im), (2,2), (3,4)) == B * (2 + 3.0im)
        @test A / (2 + 3.0im) === OneElement(val / (2 + 3.0im), (2,2), (3,4)) == B / (2 + 3.0im)
    end


    @testset "isbanded" begin
        A = OneElement(3, (2,3), (4,5))
        @test !isdiag(A)
        @test istriu(A)
        @test !istril(A)
        @test LinearAlgebra.isbanded(A, 1, 2)
        @test LinearAlgebra.isbanded(A, -1, 2)
        @test !LinearAlgebra.isbanded(A, 2, 2)

        A = OneElement(3, (4,2), (4,5))
        @test !isdiag(A)
        @test !istriu(A)
        @test istril(A)
        @test LinearAlgebra.isbanded(A, -2, -2)
        @test LinearAlgebra.isbanded(A, -2, 2)
        @test !LinearAlgebra.isbanded(A, 2, 2)

        for A in (OneElement(3, (2,2), (4,5)), OneElement(3, (1,1), (1,2)), OneElement(3, (8,7), (2,1)))
            @test isdiag(A)
            @test istriu(A)
            @test istril(A)
        end

        for A in (OneElement(0, (2,3), (4,5)), OneElement(0, (3,2), (4,5)))
            @test isdiag(A)
            @test istriu(A)
            @test istril(A)
        end
    end

    @testset "zero/iszero" begin
        v = OneElement(10, 3, 4)
        @test v + zero(v) == v
        @test typeof(zero(v)) == typeof(v)

        @test !iszero(v)
        @test iszero(OneElement(0, 3, 4))
        @test iszero(OneElement(0, 5, 4))
        @test iszero(OneElement(3, (2,2), (0,0)))
        @test iszero(OneElement(3, (2,2), (1,2)))

        v = OneElement(SMatrix{2,2}(1:4), 3, 4)
        @test v + zero(v) == v
        @test typeof(zero(v)) == typeof(v)
    end

    @testset "isone" begin
        @test isone(OneElement(3, (0,0), (0,0)))
        @test isone(OneElement(1, (1,1), (1,1)))
        @test !isone(OneElement(2, (1,1), (1,1)))
        @test !isone(OneElement(1, (2,2), (2,2)))
    end

    @testset "tril/triu" begin
        for A in (OneElement(3, (4,2), (4,5)), OneElement(3, (2,3), (4,5)), OneElement(3, (3,3), (4,5)))
            B = Array(A)
            for k in -5:5
                @test tril(A, k) == tril(B, k)
                @test triu(A, k) == triu(B, k)
            end
        end
    end

    @testset "broadcasting" begin
        for v in (OneElement(-2, 3, 4), OneElement(2im, (1,2), (3,4)))
            w = Array(v)
            n = 2
            @test abs.(v) == abs.(w)
            @test abs2.(v) == abs2.(w)
            @test real.(v) == real.(w)
            @test imag.(v) == imag.(w)
            @test conj.(v) == conj.(w)
            @test v .^ n == w .^ n
            @test v .* n == w .* n
            @test v ./ n == w ./ n
            @test n .\ v == n .\ w
        end
    end

    @testset "permutedims" begin
        v = OneElement(1, (2, 3), (2, 5))
        @test permutedims(v) === OneElement(1, (3, 2), (5, 2))
        w = OneElement(1, 3, 5)
        @test permutedims(w) === OneElement(1, (1, 3), (1, 5))
        x = OneElement(1, (1, 2, 3), (4, 5, 6))
        @test permutedims(x, (1, 2, 3)) === x
        @test permutedims(x, (3, 2, 1)) === OneElement(1, (3, 2, 1), (6, 5, 4))
        @test permutedims(x, [2, 3, 1]) === OneElement(1, (2, 3, 1), (5, 6, 4))
        @test_throws BoundsError permutedims(x, (2, 1))
    end
    @testset "show" begin
        B = OneElement(2, (1, 2), (3, 4))
        @test repr(B) == "OneElement(2, (1, 2), (3, 4))"
        B = OneElement(2, 1, 3)
        @test repr(B) == "OneElement(2, 1, 3)"
        B = OneElement(2, (1, 2), (Base.IdentityUnitRange(1:1), Base.IdentityUnitRange(2:2)))
        @test repr(B) == "OneElement(2, (1, 2), (Base.IdentityUnitRange(1:1), Base.IdentityUnitRange(2:2)))"
    end
end

@testset "repeat" begin
    @testset "0D" begin
        @test repeat(Zeros()) isa Zeros
        @test repeat(Zeros()) == repeat(zeros())
        @test repeat(Ones()) isa Ones
        @test repeat(Ones()) == repeat(ones())
        @test repeat(Fill(3)) isa Fill
        @test repeat(Fill(3)) == repeat(fill(3))

        @test repeat(Zeros(), inner=(), outer=()) isa Zeros
        @test repeat(Zeros(), inner=(), outer=()) == repeat(zeros(), inner=(), outer=())
        @test repeat(Ones(), inner=(), outer=()) isa Ones
        @test repeat(Ones(), inner=(), outer=()) == repeat(ones(), inner=(), outer=())
        @test repeat(Fill(4), inner=(), outer=()) isa Fill
        @test repeat(Fill(4), inner=(), outer=()) == repeat(fill(4), inner=(), outer=())

        @test repeat(Zeros{Bool}(), 2, 3) isa Zeros{Bool}
        @test repeat(Zeros{Bool}(), 2, 3) == repeat(zeros(Bool), 2, 3)
        @test repeat(Ones{Bool}(), 2, 3) isa Ones{Bool}
        @test repeat(Ones{Bool}(), 2, 3) == repeat(ones(Bool), 2, 3)
        @test repeat(Fill(false), 2, 3) isa Fill
        @test repeat(Fill(false), 2, 3) == repeat(fill(false), 2, 3)

        @test repeat(Zeros(), inner=(2,2), outer=5) isa Zeros
        @test repeat(Zeros(), inner=(2,2), outer=5) == repeat(zeros(), inner=(2,2), outer=5)
        @test repeat(Ones(), inner=(2,2), outer=5) isa Ones
        @test repeat(Ones(), inner=(2,2), outer=5) == repeat(ones(), inner=(2,2), outer=5)
        @test repeat(Fill(2), inner=(2,2), outer=5) isa Fill
        @test repeat(Fill(2), inner=(2,2), outer=5) == repeat(fill(2), inner=(2,2), outer=5)

        @test repeat(Zeros(), inner=(2,2), outer=(2,3)) isa Zeros
        @test repeat(Zeros(), inner=(2,2), outer=(2,3)) == repeat(zeros(), inner=(2,2), outer=(2,3))
        @test repeat(Ones(), inner=(2,2), outer=(2,3)) isa Ones
        @test repeat(Ones(), inner=(2,2), outer=(2,3)) == repeat(ones(), inner=(2,2), outer=(2,3))
        @test repeat(Fill("a"), inner=(2,2), outer=(2,3)) isa Fill
        @test repeat(Fill("a"), inner=(2,2), outer=(2,3)) == repeat(fill("a"), inner=(2,2), outer=(2,3))
    end
    @testset "1D" begin
        @test repeat(Zeros(2), 2, 3) isa Zeros
        @test repeat(Zeros(2), 2, 3) == repeat(zeros(2), 2, 3)
        @test repeat(Ones(2), 2, 3) isa Ones
        @test repeat(Ones(2), 2, 3) == repeat(ones(2), 2, 3)
        @test repeat(Fill(2,3), 2, 3) isa Fill
        @test repeat(Fill(2,3), 2, 3) == repeat(fill(2,3), 2, 3)

        @test repeat(Zeros(2), inner=2, outer=4) isa Zeros
        @test repeat(Zeros(2), inner=2, outer=4) == repeat(zeros(2), inner=2, outer=4)
        @test repeat(Ones(2), inner=2, outer=4) isa Ones
        @test repeat(Ones(2), inner=2, outer=4) == repeat(ones(2), inner=2, outer=4)
        @test repeat(Fill(2,3), inner=2, outer=4) isa Fill
        @test repeat(Fill(2,3), inner=2, outer=4) == repeat(fill(2,3), inner=2, outer=4)

        @test repeat(Zeros(2), inner=2, outer=(2,3)) isa Zeros
        @test repeat(Zeros(2), inner=2, outer=(2,3)) == repeat(zeros(2), inner=2, outer=(2,3))
        @test repeat(Ones(2), inner=2, outer=(2,3)) isa Ones
        @test repeat(Ones(2), inner=2, outer=(2,3)) == repeat(ones(2), inner=2, outer=(2,3))
        @test repeat(Fill("b",3), inner=2, outer=(2,3)) isa Fill
        @test repeat(Fill("b",3), inner=2, outer=(2,3)) == repeat(fill("b",3), inner=2, outer=(2,3))

        @test repeat(Zeros(Int, 2), inner=(2,), outer=(2,3)) isa Zeros
        @test repeat(Zeros(Int, 2), inner=(2,), outer=(2,3)) == repeat(zeros(Int, 2), inner=(2,), outer=(2,3))
        @test repeat(Ones(Int, 2), inner=(2,), outer=(2,3)) isa Ones
        @test repeat(Ones(Int, 2), inner=(2,), outer=(2,3)) == repeat(ones(Int, 2), inner=(2,), outer=(2,3))
        @test repeat(Fill(2,3), inner=(2,), outer=(2,3)) isa Fill
        @test repeat(Fill(2,3), inner=(2,), outer=(2,3)) == repeat(fill(2,3), inner=(2,), outer=(2,3))

        @test repeat(Zeros(2), inner=(2,2,1,4), outer=(2,3)) isa Zeros
        @test repeat(Zeros(2), inner=(2,2,1,4), outer=(2,3)) == repeat(zeros(2), inner=(2,2,1,4), outer=(2,3))
        @test repeat(Ones(2), inner=(2,2,1,4), outer=(2,3)) isa Ones
        @test repeat(Ones(2), inner=(2,2,1,4), outer=(2,3)) == repeat(ones(2), inner=(2,2,1,4), outer=(2,3))
        @test repeat(Fill(2,3), inner=(2,2,1,4), outer=(2,3)) isa Fill
        @test repeat(Fill(2,3), inner=(2,2,1,4), outer=(2,3)) == repeat(fill(2,3), inner=(2,2,1,4), outer=(2,3))

        @test_throws ArgumentError repeat(Fill(2,3), inner=())
        @test_throws ArgumentError repeat(Fill(2,3), outer=())
    end

    @testset "2D" begin
        @test repeat(Zeros(2,3), 2, 3) isa Zeros
        @test repeat(Zeros(2,3), 2, 3) == repeat(zeros(2,3), 2, 3)
        @test repeat(Ones(2,3), 2, 3) isa Ones
        @test repeat(Ones(2,3), 2, 3) == repeat(ones(2,3), 2, 3)
        @test repeat(Fill(2,3,4), 2, 3) isa Fill
        @test repeat(Fill(2,3,4), 2, 3) == repeat(fill(2,3,4), 2, 3)

        @test repeat(Zeros(2,3), inner=(1,2), outer=(4,2)) isa Zeros
        @test repeat(Zeros(2,3), inner=(1,2), outer=(4,2)) == repeat(zeros(2,3), inner=(1,2), outer=(4,2))
        @test repeat(Ones(2,3), inner=(1,2), outer=(4,2)) isa Ones
        @test repeat(Ones(2,3), inner=(1,2), outer=(4,2)) == repeat(ones(2,3), inner=(1,2), outer=(4,2))
        @test repeat(Fill(2,3,4), inner=(1,2), outer=(4,2)) isa Fill
        @test repeat(Fill(2,3,4), inner=(1,2), outer=(4,2)) == repeat(fill(2,3,4), inner=(1,2), outer=(4,2))

        @test repeat(Zeros(2,3), inner=(2,2,1,4), outer=(2,1,3)) isa Zeros
        @test repeat(Zeros(2,3), inner=(2,2,1,4), outer=(2,1,3)) == repeat(zeros(2,3), inner=(2,2,1,4), outer=(2,1,3))
        @test repeat(Ones(2,3), inner=(2,2,1,4), outer=(2,1,3)) isa Ones
        @test repeat(Ones(2,3), inner=(2,2,1,4), outer=(2,1,3)) == repeat(ones(2,3), inner=(2,2,1,4), outer=(2,1,3))
        @test repeat(Fill(2,3,4), inner=(2,2,1,4), outer=(2,1,3)) isa Fill
        @test repeat(Fill(2,3,4), inner=(2,2,1,4), outer=(2,1,3)) == repeat(fill(2,3,4), inner=(2,2,1,4), outer=(2,1,3))

        @test_throws ArgumentError repeat(Fill(2,3,4), inner=())
        @test_throws ArgumentError repeat(Fill(2,3,4), outer=())
        @test_throws ArgumentError repeat(Fill(2,3,4), inner=(1,))
        @test_throws ArgumentError repeat(Fill(2,3,4), outer=(1,))
    end
end

@testset "structured matrix" begin
    # strange bug on Julia v1.6, see
    # https://discourse.julialang.org/t/strange-seemingly-out-of-bounds-access-bug-in-julia-v1-6/101041
    bands = if VERSION >= v"1.9"
        ((Fill(2,3), Fill(6,2)), (Zeros(3), Zeros(2)))
    else
        ((Fill(2,3), Fill(6,2)),)
    end
    @testset for (dv, ev) in bands
        for D in (Diagonal(dv), Bidiagonal(dv, ev, :U),
                    Tridiagonal(ev, dv, ev), SymTridiagonal(dv, ev))

            M = Matrix(D)
            for k in -5:5
                @test diag(D, k) isa FillArrays.AbstractFill{eltype(D),1}
                @test diag(D, k) == diag(M, k)
            end
        end
    end
end

@testset "ReverseDiff with Zeros" begin
    # MWE in https://github.com/JuliaArrays/FillArrays.jl/issues/252
    @test ReverseDiff.gradient(x -> sum(abs2.((Zeros(5) .- zeros(5)) ./ x)), rand(5)) == zeros(5)
    @test ReverseDiff.gradient(x -> sum(abs2.((zeros(5) .- Zeros(5)) ./ x)), rand(5)) == zeros(5)
    # MWE in https://github.com/JuliaArrays/FillArrays.jl/pull/278
    @test ReverseDiff.gradient(x -> sum(abs2.((Zeros{eltype(x)}(5) .- zeros(5)) ./ x)), rand(5)) == zeros(5)
    @test ReverseDiff.gradient(x -> sum(abs2.((zeros(5) .- Zeros{eltype(x)}(5)) ./ x)), rand(5)) == zeros(5)

    # Corresponding tests with +
    @test ReverseDiff.gradient(x -> sum(abs2.((Zeros(5) .+ zeros(5)) ./ x)), rand(5)) == zeros(5)
    @test ReverseDiff.gradient(x -> sum(abs2.((zeros(5) .+ Zeros(5)) ./ x)), rand(5)) == zeros(5)
    @test ReverseDiff.gradient(x -> sum(abs2.((Zeros{eltype(x)}(5) .+ zeros(5)) ./ x)), rand(5)) == zeros(5)
    @test ReverseDiff.gradient(x -> sum(abs2.((zeros(5) .+ Zeros{eltype(x)}(5)) ./ x)), rand(5)) == zeros(5)
end

@testset "FillArraysPDMatsExt" begin
    for diag in (Ones(5), Fill(4.1, 8))
        a = @inferred(AbstractPDMat(Diagonal(diag)))
        @test a isa ScalMat
        @test a.dim == length(diag)
        @test a.value == first(diag)
    end
end

@testset "isbanded/isdiag" begin
    @testset for A in Any[Zeros(2,3), Zeros(0,1), Zeros(1,1), Zeros(1,2),
            Ones(0,1), Ones(1,1), Ones(3,4), Ones(0,4), Ones(7,0), Ones(7,2), Ones(2,7),
            Fill(3, 0,1), Fill(3, 1,1), Fill(3, 2,4), Fill(0, 3, 4), Fill(2, 0, 4), Fill(2, 6, 0), Fill(0, 8, 8)]
        B = Array(A)
        @test isdiag(A) == isdiag(B)
        for k in -5:5
            @test istriu(A, k) == istriu(B, k)
            @test istril(A, k) == istril(B, k)
        end
    end
end

@testset "triu/tril for Zeros" begin
    Z = Zeros(3, 4)
    @test triu(Z) === Z
    @test tril(Z) === Z
    @test triu(Z, 2) === Z
    @test tril(Z, 2) === Z
end
