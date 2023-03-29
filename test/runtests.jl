using FillArrays, LinearAlgebra, SparseArrays, StaticArrays, Random, Base64, Test, Statistics
import FillArrays: AbstractFill, RectDiagonal, SquareEye

using Aqua
@testset "Project quality" begin
    Aqua.test_all(FillArrays, ambiguities=false)
end

include("infinitearrays.jl")

@testset "fill array constructors and convert" begin
    for (Typ, funcs) in ((:Zeros, :zeros), (:Ones, :ones))
        @eval begin
            @test $Typ((-1,5)) == $Typ((0,5))
            @test $Typ(5) isa AbstractVector{Float64}
            @test $Typ(5,5) isa AbstractMatrix{Float64}
            @test $Typ(5) == $Typ((5,))
            @test $Typ(5,5) == $Typ((5,5))
            @test eltype($Typ(5,5)) == Float64

            for T in (Int, Float64)
                Z = $Typ{T}(5)
                @test $Typ(T, 5) ≡ Z
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
                @test $Typ(T, 5, 5) ≡ Z
                @test eltype(Z) == T
                @test Array(Z) == $funcs(T,5,5)
                @test Array{T}(Z) == $funcs(T,5,5)
                @test Array{T,2}(Z) == $funcs(T,5,5)

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

                @test $Typ{T,2}(2ones(T,5,5)) ≡ $Typ{T}(5,5)
                @test $Typ{T}(2ones(T,5,5)) ≡ $Typ{T}(5,5)
                @test $Typ(2ones(T,5,5)) ≡ $Typ{T}(5,5)

                @test $Typ(Z) ≡ $Typ{T}(Z) ≡ $Typ{T,2}(Z) ≡ typeof(Z)(Z) ≡ Z

                @test AbstractArray{Float32}(Z) ≡ $Typ{Float32}(5,5)
                @test AbstractArray{Float32,2}(Z) ≡ $Typ{Float32}(5,5)
            end
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
function test_addition_and_subtraction(As, Bs, Tout::Type)
    for A in As, B in Bs
        @testset "$(typeof(A)) ± $(typeof(B))" begin
            @test A + B isa Tout{promote_type(eltype(A), eltype(B))}
            @test as_array(A + B) == as_array(A) + as_array(B)

            @test A - B isa Tout{promote_type(eltype(A), eltype(B))}
            @test as_array(A - B) == as_array(A) - as_array(B)

            @test B + A isa Tout{promote_type(eltype(B), eltype(A))}
            @test as_array(B + A) == as_array(B) + as_array(A)

            @test B - A isa Tout{promote_type(eltype(B), eltype(A))}
            @test as_array(B - A) == as_array(B) - as_array(A)
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
    test_addition_and_subtraction((A_fill, B_fill), (A_fill, B_fill), Fill)
    test_addition_and_subtraction_dim_mismatch(A_fill, Fill(randn(rng), 5, 2))

    # FillArray + Array (etc) should construct a new Array using `getindex`.
    A_dense, B_dense = randn(rng, 5), [5, 4, 3, 2, 1]
    test_addition_and_subtraction((A_fill, B_fill), (A_dense, B_dense), Array)
    test_addition_and_subtraction_dim_mismatch(A_fill, randn(rng, 5, 2))

    # FillArray + StepLenRange / UnitRange (etc) should yield an AbstractRange.
    A_ur, B_ur = 1.0:5.0, 6:10
    test_addition_and_subtraction((A_fill, B_fill), (A_ur, B_ur), AbstractRange)
    test_addition_and_subtraction_dim_mismatch(A_fill, 1.0:6.0)
    test_addition_and_subtraction_dim_mismatch(A_fill, 5:10)

    # FillArray + UniformScaling should yield a Matrix in general
    As_fill_square = (Fill(randn(rng, Float64), 3, 3), Fill(5, 4, 4))
    Bs_us = (UniformScaling(2.3), UniformScaling(3))
    test_addition_and_subtraction(As_fill_square, Bs_us, Matrix)
    As_fill_nonsquare = (Fill(randn(rng, Float64), 3, 2), Fill(5, 3, 4))
    for A in As_fill_nonsquare, B in Bs_us
        test_addition_and_subtraction_dim_mismatch(A, B)
    end

    # FillArray + StaticArray should not have ambiguities
    A_svec, B_svec = SVector{5}(rand(5)), SVector(1, 2, 3, 4, 5)
    test_addition_and_subtraction((A_fill, B_fill, Zeros(5)), (A_svec, B_svec), SVector{5})

    # Issue #224
    A_matmat, B_matmat = Fill(rand(3,3),5), [rand(3,3) for n=1:5]
    test_addition_and_subtraction((A_matmat,), (A_matmat,), Fill)
    test_addition_and_subtraction((B_matmat,), (A_matmat,), Vector)

    # Optimizations for Zeros and RectOrDiagonal{<:Any, <:AbstractFill}
    As_special_square = (
        Zeros(3, 3), Zeros{Int}(4, 4),
        Eye(3), Eye{Int}(4), Eye(3, 3), Eye{Int}(4, 4),
        Diagonal(Fill(randn(rng, Float64), 3)), Diagonal(Fill(3, 4)),
        RectDiagonal(Fill(randn(rng, Float64), 3), 3, 3), RectDiagonal(Fill(3, 4), 4, 4)
    )
    DiagonalAbstractFill{T} = Diagonal{T, <:AbstractFill{T, 1}}
    test_addition_and_subtraction(As_special_square, Bs_us, DiagonalAbstractFill)
    As_special_nonsquare = (
        Zeros(3, 2), Zeros{Int}(3, 4),
        Eye(3, 2), Eye{Int}(3, 4),
        RectDiagonal(Fill(randn(rng, Float64), 2), 3, 2), RectDiagonal(Fill(3, 3), 3, 4)
    )
    for A in As_special_nonsquare, B in Bs_us
        test_addition_and_subtraction_dim_mismatch(A, B)
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

    @testset "Check multiplication by Adjoint vectors works as expected." begin
        @test randn(4, 3)' * Zeros(4) ≡ Zeros(3)
        @test randn(4)' * Zeros(4) ≡ transpose(randn(4)) * Zeros(4) ≡ zero(Float64)
        @test [1, 2, 3]' * Zeros{Int}(3) ≡ zero(Int)
        @test [SVector(1,2)', SVector(2,3)', SVector(3,4)']' * Zeros{Int}(3) === SVector(0,0)
        @test_throws DimensionMismatch randn(4)' * Zeros(3)
        @test Zeros(5)' * randn(5,3) ≡ Zeros(5)'*Zeros(5,3) ≡ Zeros(5)'*Ones(5,3) ≡ Zeros(3)'
        @test Zeros(5)' * randn(5) ≡ Zeros(5)' * Zeros(5) ≡ Zeros(5)' * Ones(5) ≡ 0.0
        @test Zeros(5) * Zeros(6)' ≡ Zeros(5,1) * Zeros(6)' ≡ Zeros(5,6)
        @test randn(5) * Zeros(6)' ≡ randn(5,1) * Zeros(6)' ≡ Zeros(5,6)
        @test Zeros(5) * randn(6)' ≡ Zeros(5,6)

        @test ([[1,2]])' * Zeros{SVector{2,Int}}(1) ≡ 0
        @test_broken ([[1,2,3]])' * Zeros{SVector{2,Int}}(1)
    end

    @testset "Check multiplication by Transpose-d vectors works as expected." begin
        @test transpose(randn(4, 3)) * Zeros(4) === Zeros(3)
        @test transpose(randn(4)) * Zeros(4) === zero(Float64)
        @test transpose([1, 2, 3]) * Zeros{Int}(3) === zero(Int)
        @test_throws DimensionMismatch transpose(randn(4)) * Zeros(3)
        @test transpose(Zeros(5)) * randn(5,3) ≡ transpose(Zeros(5))*Zeros(5,3) ≡ transpose(Zeros(5))*Ones(5,3) ≡ transpose(Zeros(3))
        @test transpose(Zeros(5)) * randn(5) ≡ transpose(Zeros(5)) * Zeros(5) ≡ transpose(Zeros(5)) * Ones(5) ≡ 0.0
        @test randn(5) * transpose(Zeros(6)) ≡ randn(5,1) * transpose(Zeros(6)) ≡ Zeros(5,6)
        @test Zeros(5) * transpose(randn(6)) ≡ Zeros(5,6)
        @test transpose(randn(5)) * Zeros(5) ≡ 0.0

        @test transpose([[1,2]]) * Zeros{SVector{2,Int}}(1) ≡ 0
        @test_broken transpose([[1,2,3]]) * Zeros{SVector{2,Int}}(1)
    end

    z1, z2 = Zeros{Float64}(4), Zeros{Int}(4)

    @testset "`Zeros` are closed under addition and subtraction (both unary and binary)." begin
        @test +(Zeros{Float64}(3, 5)) === Zeros{Float64}(3, 5)
        @test -(Zeros{Float32}(5, 2)) === Zeros{Float32}(5, 2)

        @test +(z1) === z1
        @test -(z1) === z1

        test_addition_and_subtraction((z1, z2), (z1, z2), Zeros)
        test_addition_and_subtraction_dim_mismatch(z1, Zeros{Float64}(4, 2))
    end

    # `Zeros` +/- `Fill`s should yield `Fills`.
    fill1, fill2 = Fill(5.0, 4), Fill(5, 4)
    test_addition_and_subtraction((z1, z2), (fill1, fill2), Fill)
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
        @test !(Zeros(5) + X === X)
        @test Zeros{Int}(5) + (1:5) === (1:5) && (1:5) + Zeros{Int}(5) === (1:5)
        @test Zeros(5) + (1:5) === (1.0:1.0:5.0) && (1:5) + Zeros(5) === (1.0:1.0:5.0)
        @test (1:5) - Zeros{Int}(5) === (1:5)
        @test Zeros{Int}(5) - (1:5) === -1:-1:-5
        @test Zeros(5) - (1:5) === -1.0:-1.0:-5.0
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
end

@testset "Cumsum and diff" begin
    @test sum(Fill(3,10)) ≡ 30
    @test reduce(+, Fill(3,10)) ≡ 30
    @test sum(x -> x + 1, Fill(3,10)) ≡ 40
    @test cumsum(Fill(3,10)) ≡ StepRangeLen(3,3,10)

    @test sum(Ones(10)) ≡ 10.0
    @test sum(x -> x + 1, Ones(10)) ≡ 20.0
    @test cumsum(Ones(10)) ≡ StepRangeLen(1.0, 1.0, 10)

    @test sum(Ones{Int}(10)) ≡ 10
    @test sum(x -> x + 1, Ones{Int}(10)) ≡ 20
    @test cumsum(Ones{Int}(10)) ≡ Base.OneTo(10)

    @test sum(Zeros(10)) ≡ 0.0
    @test sum(x -> x + 1, Zeros(10)) ≡ 10.0
    @test cumsum(Zeros(10)) ≡ Zeros(10)

    @test sum(Zeros{Int}(10)) ≡ 0
    @test sum(x -> x + 1, Zeros{Int}(10)) ≡ 10
    @test cumsum(Zeros{Int}(10)) ≡ Zeros{Int}(10)

    @test cumsum(Zeros{Bool}(10)) ≡ Zeros{Bool}(10)
    @test cumsum(Ones{Bool}(10)) ≡ Base.OneTo{Int}(10)
    @test cumsum(Fill(true,10)) ≡ StepRangeLen(true, true, 10)

    @test diff(Fill(1,10)) ≡ Zeros{Int}(9)
    @test diff(Ones{Float64}(10)) ≡ Zeros{Float64}(9)
    @test_throws UndefKeywordError cumsum(Fill(1,1,5))

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

        @testset "Addition" begin
            @test Zeros{Int}(5) .+ (1:5) ≡ (1:5) .+ Zeros{Int}(5) ≡ (1:5) .- Zeros{Int}(5) ≡ 1:5
            @test Zeros{Int}(1) .+ (1:5) ≡ (1:5) .+ Zeros{Int}(1) ≡ (1:5) .- Zeros{Int}(1) ≡ 1:5
            @test Zeros(5) .+ (1:5) == (1:5) .+ Zeros(5) == (1:5) .- Zeros(5) == 1:5
            @test Zeros{Int}(5) .+ Fill(1,5) ≡ Fill(1,5) .+ Zeros{Int}(5) ≡ Fill(1,5) .- Zeros{Int}(5) ≡ Fill(1,5)
            @test_throws DimensionMismatch Zeros{Int}(2) .+ (1:5)
            @test_throws DimensionMismatch (1:5) .+ Zeros{Int}(2)
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

    @testset "Zero .*" begin
        @test Zeros{Int}(10) .* Zeros{Int}(10) ≡ Zeros{Int}(10)
        @test randn(10) .* Zeros(10) ≡ Zeros(10)
        @test Zeros(10) .* randn(10) ≡ Zeros(10)
        @test (1:10) .* Zeros(10) ≡ Zeros(10)
        @test Zeros(10) .* (1:10) ≡ Zeros(10)
        @test_throws DimensionMismatch (1:11) .* Zeros(10)
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
    @test map(() -> "ok") == "ok"  # was MethodError: reducing over an empty collection is not allowed
    @test mapreduce(() -> "ok", *) == "ok"
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
        for d in (1, )
            for m in (Eye{T}(d), Eye{T}(d, d))
                @test ! any(iszero, m)
                @test ! all(iszero, m)
                @test any(isone, m)
                @test all(isone, m)
            end

            for m in (Eye{T}(d, d + 1), Eye{T}(d + 1, d))
                @test any(iszero, m)
                @test ! all(iszero, m)
                @test any(isone, m)
                @test ! all(isone, m)
            end

            onem = Ones{T}(d, d)
            @test isone(onem)
            @test ! iszero(onem)

            zerom = Zeros{T}(d, d)
            @test ! isone(zerom)
            @test  iszero(zerom)

            fillm0 = Fill(T(0), d, d)
            @test ! isone(fillm0)
            @test   iszero(fillm0)

            fillm1 = Fill(T(1), d, d)
            @test isone(fillm1)
            @test ! iszero(fillm1)

            fillm2 = Fill(T(2), d, d)
            @test ! isone(fillm2)
            @test ! iszero(fillm2)
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

    @testset "all/any" begin
        @test any(Ones{Bool}(10)) === all(Ones{Bool}(10)) === any(Fill(true,10)) === all(Fill(true,10)) === true
        @test any(Zeros{Bool}(10)) === all(Zeros{Bool}(10)) === any(Fill(false,10)) === all(Fill(false,10)) === false
        @test all(b -> ndims(b) ==  1, Fill([1,2],10))
        @test any(b -> ndims(b) ==  1, Fill([1,2],10))

        @test all(Fill(2,0))
        @test !any(Fill(2,0))
        @test any(Trues(2,0)) == any(trues(2,0))
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
    @test FillArrays._copy_oftype(m, Int) ≡ Eye{Int}(10)
    @test FillArrays._copy_oftype(D, Float64) ≡ Diagonal(Fill(2.0,10))
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

    a = randn(3)
    A = randn(1,4)

    @test Fill(2,3)*A ≈ Vector(Fill(2,3))*A
    @test Fill(2,3,1)*A ≈ Matrix(Fill(2,3,1))*A
    @test Fill(2,3,3)*a ≈ Matrix(Fill(2,3,3))*a
    @test Ones(3)*A ≈ Vector(Ones(3))*A
    @test Ones(3,1)*A ≈ Matrix(Ones(3,1))*A
    @test Ones(3,3)*a ≈ Matrix(Ones(3,3))*a
    @test Zeros(3)*A  ≡ Zeros(3,4)
    @test Zeros(3,1)*A == Zeros(3,4)
    @test Zeros(3,3)*a == Zeros(3)

    @test A*Fill(2,4) ≈ A*Vector(Fill(2,4))
    @test A*Fill(2,4,1) ≈ A*Matrix(Fill(2,4,1))
    @test a*Fill(2,1,3) ≈ a*Matrix(Fill(2,1,3))
    @test A*Ones(4) ≈ A*Vector(Ones(4))
    @test A*Ones(4,1) ≈ A*Matrix(Ones(4,1))
    @test a*Ones(1,3) ≈ a*Matrix(Ones(1,3))
    @test A*Zeros(4)  ≡ Zeros(1)
    @test A*Zeros(4,1) ≡ Zeros(1,1)
    @test a*Zeros(1,3) ≡ Zeros(3,3)

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
    @test_skip @test_throws DimensionMismatch Diagonal(Fill(1,1)) * Ones(10)
    @test_throws DimensionMismatch Diagonal(Fill(1,1)) * Ones(10,5)
    @test_throws DimensionMismatch Ones(5,10) * Diagonal(Fill(1,1))

    E = Eye(5)
    @test E*(1:5) ≡ 1.0:5.0
    @test (1:5)'E == (1.0:5)'
    @test E*E ≡ E

    for T in (Float64, ComplexF64)
        fv = T == Float64 ? Float64(1.6) : ComplexF64(1.6, 1.3)
        n  = 10
        k  = 12
        m  = 15
        fillvec = Fill(fv, k)
        fillmat = Fill(fv, k, m)
        A  = rand(ComplexF64, n, k)
        @test A*fillvec ≈ A*Array(fillvec)
        @test A*fillmat ≈ A*Array(fillmat)
        A  = rand(ComplexF64, k, n)
        @test transpose(A)*fillvec ≈ transpose(A)*Array(fillvec)
        @test transpose(A)*fillmat ≈ transpose(A)*Array(fillmat)
        @test adjoint(A)*fillvec ≈ adjoint(A)*Array(fillvec)
        @test adjoint(A)*fillmat ≈ adjoint(A)*Array(fillmat)
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

@testset "dot products" begin
    n = 15
    o = Ones(1:n)
    z = Zeros(1:n)
    D = Diagonal(o)
    Z = Diagonal(z)

    Random.seed!(5)
    u = rand(n)
    v = rand(n)

    @test dot(u, D, v) == dot(u, v)
    @test dot(u, 2D, v) == 2dot(u, v)
    @test dot(u, Z, v) == 0

    @test dot(Zeros(5), Zeros{ComplexF16}(5)) ≡ zero(ComplexF64)
    @test dot(Zeros(5), Ones{ComplexF16}(5)) ≡ zero(ComplexF64)
    @test dot(Ones{ComplexF16}(5), Zeros(5)) ≡ zero(ComplexF64)
    @test dot(randn(5), Zeros{ComplexF16}(5)) ≡ dot(Zeros{ComplexF16}(5), randn(5)) ≡ zero(ComplexF64)

    @test dot(Fill(1,5), Fill(2.0,5)) ≡ 10.0

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
    @test repr(Zeros(3)) == "Zeros(3)"
    @test repr(Ones(3,2)) == "Ones(3, 2)"
    @test repr(Fill(7,3,2)) == "Fill(7, 3, 2)"
    @test repr(Fill(1f0,10)) == "Fill(1.0f0, 10)"  # Float32!
    @test repr(Fill(0)) == "Fill(0)"
    @test repr(Eye(9)) == "Eye(9)"
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
    e₁ = OneElement(2, 5)
    @test e₁ == [0,1,0,0,0]
    @test_throws BoundsError e₁[6]

    e₁ = OneElement{Float64}(2, 5)
    @test e₁ == [0,1,0,0,0]

    v = OneElement{Float64}(2, 3, 4)
    @test v == [0,0,2,0]

    V = OneElement(2, (2,3), (3,4))
    @test V == [0 0 0 0; 0 0 2 0; 0 0 0 0]

    @test stringmime("text/plain", V) == "3×4 OneElement{$Int, 2, Tuple{$Int, $Int}, Tuple{OneTo{$Int}, OneTo{$Int}}}:\n ⋅  ⋅  ⋅  ⋅\n ⋅  ⋅  2  ⋅\n ⋅  ⋅  ⋅  ⋅"

    @test Base.setindex(Zeros(5), 2, 2) ≡ OneElement(2.0, 2, 5)
    @test Base.setindex(Zeros(5,3), 2, 2, 3) ≡ OneElement(2.0, (2,3), (5,3))
    @test_throws BoundsError Base.setindex(Zeros(5), 2, 6)
end