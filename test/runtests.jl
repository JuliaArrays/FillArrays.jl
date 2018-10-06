
using FillArrays, LinearAlgebra, SparseArrays, Random, Test

import FillArrays: AbstractFill

@testset "fill array constructors and convert" begin
    for (Typ, funcs) in ((:Zeros, :zeros), (:Ones, :ones))
        @eval begin
            @test_throws BoundsError $Typ((-1,5))
            @test $Typ(5) isa AbstractVector{Float64}
            @test $Typ(5,5) isa AbstractMatrix{Float64}
            @test $Typ(5) == $Typ((5,))
            @test $Typ(5,5) == $Typ((5,5))
            @test eltype($Typ(5,5)) == Float64

            for T in (Int, Float64)
                Z = $Typ{T}(5)
                @test eltype(Z) == T
                @test Array(Z) == $funcs(T,5)
                @test Array{T}(Z) == $funcs(T,5)
                @test Array{T,1}(Z) == $funcs(T,5)

                @test convert(AbstractArray,Z) ≡ Z
                @test convert(AbstractArray{T},Z) ≡ AbstractArray{T}(Z) ≡ Z
                @test convert(AbstractVector{T},Z) ≡ AbstractVector{T}(Z) ≡ Z

                @test $Typ{T,1}(2ones(T,5)) == Z
                @test $Typ{T}(2ones(T,5)) == Z
                @test $Typ(2ones(T,5)) == Z

                Z = $Typ{T}(5, 5)
                @test eltype(Z) == T
                @test Array(Z) == $funcs(T,5,5)
                @test Array{T}(Z) == $funcs(T,5,5)
                @test Array{T,2}(Z) == $funcs(T,5,5)

                @test convert(AbstractArray,Z) ≡ convert(AbstractFill,Z) ≡ Z
                @test convert(AbstractArray{T},Z) ≡ convert(AbstractFill{T},Z) ≡ AbstractArray{T}(Z) ≡ Z
                @test convert(AbstractMatrix{T},Z) ≡ convert(AbstractFill{T,2},Z) ≡ AbstractMatrix{T}(Z) ≡ Z


                @test $Typ{T,2}(2ones(T,5,5)) == Z
                @test $Typ{T}(2ones(T,5,5)) == Z
                @test $Typ(2ones(T,5,5)) == Z

                @test AbstractArray{Float32}(Z) == $Typ{Float32}(5,5)
                @test AbstractArray{Float32,2}(Z) == $Typ{Float32}(5,5)
            end
        end
    end

    @test_throws BoundsError Fill(1,(-1,5))
    @test Fill(1.0,5) isa AbstractVector{Float64}
    @test Fill(1.0,5,5) isa AbstractMatrix{Float64}
    @test Fill(1,5) == Fill(1,(5,))
    @test Fill(1,5,5) == Fill(1,(5,5))
    @test eltype(Fill(1.0,5,5)) == Float64

    @test Matrix{Float64}(Zeros{ComplexF64}(10,10)) == zeros(10,10)
    @test_throws InexactError Matrix{Float64}(Fill(1.0+1.0im,10,10))


    for T in (Int, Float64)
        F = Fill{T}(one(T), 5)

        @test eltype(F) == T
        @test Array(F) == fill(one(T),5)
        @test Array{T}(F) == fill(one(T),5)
        @test Array{T,1}(F) == fill(one(T),5)

        F = Fill{T}(one(T), 5, 5)
        @test eltype(F) == T
        @test Array(F) == fill(one(T),5,5)
        @test Array{T}(F) == fill(one(T),5,5)
        @test Array{T,2}(F) == fill(one(T),5,5)

        @test convert(AbstractArray,F) ≡ F
        @test convert(AbstractArray{T},F) ≡ AbstractArray{T}(F) ≡ F
        @test convert(AbstractMatrix{T},F) ≡ AbstractMatrix{T}(F) ≡ F


        @test convert(AbstractArray{Float32},F) == AbstractArray{Float32}(F) ==
                Fill{Float32}(one(Float32),5,5)
        @test convert(AbstractMatrix{Float32},F) == AbstractMatrix{Float32}(F) ==
                Fill{Float32}(one(Float32),5,5)
    end

    @test Eye(5) isa Diagonal{Float64}
    @test Eye(5) == Eye{Float64}(5)
    @test eltype(Eye(5)) == Float64

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
        @test x + Zeros{Bool}(5) ≡ x
        @test x - Zeros{Bool}(5) ≡ x
        @test Zeros{Bool}(5) + x ≡ x
        @test -x ≡ Fill(-1,5)
    end
end

# Check that all pair-wise combinations of + / - elements of As and Bs yield the correct
# type, and produce numerically correct results.
function test_addition_and_subtraction(As, Bs, Tout::Type)
    for A in As, B in Bs
        @test A + B isa Tout{promote_type(eltype(A), eltype(B))}
        @test Array(A + B) == Array(A) + Array(B)

        @test A - B isa Tout{promote_type(eltype(A), eltype(B))}
        @test Array(A - B) == Array(A) - Array(B)

        @test B + A isa Tout{promote_type(eltype(B), eltype(A))}
        @test Array(B + A) == Array(B) + Array(A)

        @test B - A isa Tout{promote_type(eltype(B), eltype(A))}
        @test Array(B - A) == Array(B) - Array(A)
    end
end

# Check that all permutations of + / - throw a `DimensionMismatch` exception.
function test_addition_and_subtraction_dim_mismatch(a, b)
    @test_throws DimensionMismatch a + b
    @test_throws DimensionMismatch a - b
    @test_throws DimensionMismatch b + a
    @test_throws DimensionMismatch b - a
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
    test_addition_and_subtraction([A_fill, B_fill], [A_fill, B_fill], Fill)
    test_addition_and_subtraction_dim_mismatch(A_fill, Fill(randn(rng), 5, 2))

    # FillArray + Array (etc) should construct a new Array using `getindex`.
    A_dense, B_dense = randn(rng, 5), [5, 4, 3, 2, 1]
    test_addition_and_subtraction([A_fill, B_fill], [A_dense, B_dense], Array)
    test_addition_and_subtraction_dim_mismatch(A_fill, randn(rng, 5, 2))

    # FillArray + StepLenRange / UnitRange (etc) should yield an AbstractRange.
    A_ur, B_ur = 1.0:5.0, 6:10
    test_addition_and_subtraction([A_fill, B_fill], (A_ur, B_ur), AbstractRange)
    test_addition_and_subtraction_dim_mismatch(A_fill, 1.0:6.0)
    test_addition_and_subtraction_dim_mismatch(A_fill, 5:10)
end

@testset "Other matrix types" begin
    @test Diagonal(Zeros(5)) == Diagonal(zeros(5))

    @test Diagonal(Zeros(8,5)) == Diagonal(zeros(5))
    @test convert(Diagonal, Zeros(5,5)) == Diagonal(zeros(5))
    @test_throws BoundsError convert(Diagonal, Zeros(8,5))

    @test convert(Diagonal{Int}, Zeros(5,5)) == Diagonal(zeros(Int,5))
    @test_throws BoundsError convert(Diagonal{Int}, Zeros(8,5))


    @test convert(Diagonal, Eye(5)) == Diagonal(ones(5))
    @test convert(Diagonal{Int}, Eye(5)) == Diagonal(ones(Int,5))
end

@testset "Sparse vectors and matrices" begin
    @test SparseVector(Zeros(5)) ==
            SparseVector{Float64}(Zeros(5)) ==
            SparseVector{Float64,Int}(Zeros(5)) ==
            convert(AbstractSparseArray,Zeros(5)) ==
            convert(AbstractSparseVector,Zeros(5)) ==
            convert(AbstractSparseArray{Float64},Zeros(5)) ==
            convert(AbstractSparseVector{Float64},Zeros(5)) ==
            convert(AbstractSparseVector{Float64,Int},Zeros(5)) ==
            spzeros(5)

    for (Mat, SMat) in ((Zeros(5,5), spzeros(5,5)), (Zeros(6,5), spzeros(6,5)),
                        (Eye(5), sparse(I,5,5)))
        @test SparseMatrixCSC(Mat) ==
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
end

@testset "Rank" begin
    @test rank(Zeros(5,4)) == 0
    @test rank(Ones(5,4)) == 1
    @test rank(Fill(2,5,4)) == 1
    @test rank(Fill(0,5,4)) == 0
    @test rank(Eye(2)) == 2
end

@testset "BigInt indices" begin
    for A in (Zeros(BigInt(100)), Ones(BigInt(100)), Fill(2, BigInt(100)))
        @test length(A) isa BigInt
        @test axes(A) == tuple(Base.OneTo{BigInt}(BigInt(100)))
        @test size(A) isa Tuple{BigInt}
    end
    let A = Eye(BigInt(100))
        @test length(A) isa BigInt
        @test axes(A) == tuple(Base.OneTo{BigInt}(BigInt(100)),Base.OneTo{BigInt}(BigInt(100)))
        @test size(A) isa Tuple{BigInt,BigInt}
    end
    for A in (Zeros(BigInt(10), 10), Ones(BigInt(10), 10), Fill(2.0, (BigInt(10), 10)))
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

    @test [1,2,3]*Zeros(1) ≡ Zeros(3)
    @test [1,2,3]*Zeros(1,3) ≡ Zeros(3,3)
    @test_throws DimensionMismatch [1,2,3]*Zeros(3)

    # Check multiplication by Adjoint vectors works as expected.
    @test randn(4, 3)' * Zeros(4) === Zeros(3)
    @test randn(4)' * Zeros(4) === zero(Float64)
    @test [1, 2, 3]' * Zeros{Int}(3) === zero(Int)
    @test_throws DimensionMismatch randn(4)' * Zeros(3)

    # Check multiplication by Transpose-d vectors works as expected.
    @test transpose(randn(4, 3)) * Zeros(4) === Zeros(3)
    @test transpose(randn(4)) * Zeros(4) === zero(Float64)
    @test transpose([1, 2, 3]) * Zeros{Int}(3) === zero(Int)
    @test_throws DimensionMismatch transpose(randn(4)) * Zeros(3)

    @test +(Zeros{Float64}(3, 5)) === Zeros{Float64}(3, 5)
    @test -(Zeros{Float32}(5, 2)) === Zeros{Float32}(5, 2)

    # `Zeros` are closed under addition and subtraction (both unary and binary).
    z1, z2 = Zeros{Float64}(4), Zeros{Int}(4)
    @test +(z1) === z1
    @test -(z1) === z1

    test_addition_and_subtraction([z1, z2], [z1, z2], Zeros)
    test_addition_and_subtraction_dim_mismatch(z1, Zeros{Float64}(4, 2))

    # `Zeros` +/- `Fill`s should yield `Fills`.
    fill1, fill2 = Fill(5.0, 4), Fill(5, 4)
    test_addition_and_subtraction([z1, z2], [fill1, fill2], Fill)
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

    # Zeros +/- dense where + / - have different results.
    @test +(Zeros(3, 5), X) == X && +(X, Zeros(3, 5)) == X
    @test !(Zeros(3, 5) + X === X) && !(X + Zeros(3, 5) === X)
    @test -(Zeros(3, 5), X) == -X

    # Addition with different eltypes.
    @test +(Zeros{Float32}(3, 5), X) isa Matrix{Float64}
    @test !(+(Zeros{Float32}(3, 5), X) === X)
    @test +(Zeros{Float32}(3, 5), X) == X
    @test !(+(Zeros{ComplexF64}(3, 5), X) === X)
    @test +(Zeros{ComplexF64}(3, 5), X) == X

    # Subtraction with different eltypes.
    @test -(Zeros{Float32}(3, 5), X) isa Matrix{Float64}
    @test -(Zeros{Float32}(3, 5), X) == -X
    @test -(Zeros{ComplexF64}(3, 5), X) == -X

    # Tests for ranges.
    X = randn(5)
    @test !(Zeros(5) + X === X)
    @test Zeros{Int}(5) + (1:5) === (1:5) && (1:5) + Zeros{Int}(5) === (1:5)
    @test Zeros(5) + (1:5) === (1.0:1.0:5.0) && (1:5) + Zeros(5) === (1.0:1.0:5.0)
    @test (1:5) - Zeros{Int}(5) === (1:5)
    @test Zeros{Int}(5) - (1:5) === -1:-1:-5
    @test Zeros(5) - (1:5) === -1.0:-1.0:-5.0
end

@testset "maximum/minimum/svd/sort" begin
    @test maximum(Fill(1, 1_000_000_000)) == minimum(Fill(1, 1_000_000_000)) == 1
    @test svdvals(fill(2,5,6)) ≈ svdvals(Fill(2,5,6))
    @test svdvals(Eye(5)) === Ones(5)
    @test sort(Ones(5)) == sort!(Ones(5))
end

@testset "Cumsum" begin
    @test sum(Fill(3,10)) ≡ 30
    @test cumsum(Fill(3,10)) ≡ 3:3:30

    @test sum(Ones(10)) ≡ 10.0
    @test cumsum(Ones(10)) ≡ 1.0:10.0

    @test sum(Ones{Int}(10)) ≡ 10
    @test cumsum(Ones{Int}(10)) ≡ Base.OneTo(10)

    @test sum(Zeros(10)) ≡ 0.0
    @test cumsum(Zeros(10)) ≡ Zeros(10)

    @test sum(Zeros{Int}(10)) ≡ 0
    @test cumsum(Zeros{Int}(10)) ≡ Zeros{Int}(10)

    @test cumsum(Zeros{Bool}(10)) ≡ Zeros{Bool}(10)
    @test cumsum(Ones{Bool}(10)) ≡ Base.OneTo{Int}(10)
    @test cumsum(Fill(true,10)) ≡ 1:1:10
end

@testset "Broadcast" begin
    x = Fill(5,5)
    @test (.+)(x) ≡ x
    @test (.-)(x) ≡ -x
    @test exp.(x) ≡ Fill(exp(5),5)
    @test x .+ 1 ≡ Fill(6,5)
    @test x .+ x ≡ Fill(10,5)
    @test x .+ Ones(5) ≡ Fill(6.0,5)
    f = (x,y) -> cos(x*y)
    @test f.(x, Ones(5)) ≡ Fill(f(5,1.0),5)

    y = Ones(5,5)
    @test (.+)(y) ≡ Fill(1.0,5,5)
    @test (.-)(y) ≡ Fill(-1.0,5,5)
    @test exp.(y) ≡ Fill(exp(1),5,5)
    @test y .+ 1 ≡ Fill(2.0,5,5)
    @test y .+ y ≡ Fill(2.0,5,5)
    @test y .* y ≡ y ./ y ≡ y .\ y ≡ y

    @test Zeros{Int}(5) .+ Zeros(5) isa Zeros{Float64}
end

@testset "Sub-arrays" begin
    A = Fill(3.0,5)
    @test A[1:3] ≡ Fill(3.0,3)
    @test A[1:3,1:1] ≡ Fill(3.0,3,1)
    @test_broken A[1:3,2] ≡ Zeros{Int}(3)
    @test_throws BoundsError A[1:26]
    @test A[[true, false, true, false, false]] ≡ Fill(3.0, 2)
    A = Fill(3.0, 2, 2)
    @test A[[true true; true false]] ≡ Fill(3.0, 3)
    @test_throws DimensionMismatch A[[true, false]]

    A = Ones{Int}(5,5)
    @test A[1:3] ≡ Ones{Int}(3)
    @test A[1:3,1:2] ≡ Ones{Int}(3,2)
    @test_broken A[1:3,2] ≡ Ones{Int}(3)
    @test_throws BoundsError A[1:26]
    A = Ones{Int}(2,2)
    @test A[[true false; true false]] ≡ Ones{Int}(2)
    @test A[[true, false, true, false]] ≡ Ones{Int}(2)
    @test_throws DimensionMismatch A[[true false false; true false false]]

    A = Zeros{Int}(5,5)
    @test A[1:3] ≡ Zeros{Int}(3)
    @test A[1:3,1:2] ≡ Zeros{Int}(3,2)
    @test_broken A[1:3,2] ≡ Zeros{Int}(3)
    @test_throws BoundsError A[1:26]
    A = Zeros{Int}(2,2)
    @test A[[true false; true false]] ≡ Zeros{Int}(2)
    @test A[[true, false, true, false]] ≡ Zeros{Int}(2)
    @test_throws DimensionMismatch A[[true false false; true false false]]
end
