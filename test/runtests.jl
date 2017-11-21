using Compat
using FillArrays, Compat.Test


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

            @test AbstractArray(Z) === Z
            @test AbstractArray{T}(Z) === Z
            @test AbstractArray{T,1}(Z) === Z

            @test $Typ{T,1}(2ones(T,5)) == Z
            @test $Typ{T}(2ones(T,5)) == Z
            @test $Typ(2ones(T,5)) == Z

            Z = $Typ{T}(5, 5)
            @test eltype(Z) == T
            @test Array(Z) == $funcs(T,5,5)
            @test Array{T}(Z) == $funcs(T,5,5)
            @test Array{T,2}(Z) == $funcs(T,5,5)

            @test AbstractArray(Z) === Z
            @test AbstractArray{T}(Z) === Z
            @test AbstractArray{T,2}(Z) === Z

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

@test Matrix{Float64}(Zeros{Complex128}(10,10)) == zeros(10,10)
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

    @test AbstractArray(F) === F
    @test AbstractArray{T}(F) === F
    @test AbstractArray{T,2}(F) === F


    @test AbstractArray{Float32}(F) == Fill{Float32}(one(Float32),5,5)
    @test AbstractArray{Float32,2}(F) == Fill{Float32}(one(Float32),5,5)
end




@test_throws BoundsError Eye((-1,5))
@test Eye(5,5) isa AbstractMatrix{Float64}
@test Eye(5,5) == Eye(5) == Eye((5,5)) == Eye{Float64}(5) == Eye{Float64}(5, 5)
@test eltype(Eye(5,5)) == Float64


for T in (Int, Float64)
    E = Eye{T}(5, 5)
    M = VERSION < v"0.7.0-DEV.2565" ? eye(T, 5, 5) : Matrix{T}(I, 5, 5)

    @test eltype(E) == T
    @test Array(E) == M
    @test Array{T}(E) == M
    @test Array{T,2}(E) == M

    @test AbstractArray(E) === E
    @test AbstractArray{T}(E) === E
    @test AbstractArray{T,2}(E) === E


    @test AbstractArray{Float32}(E) == Eye{Float32}(5,5)
    @test AbstractArray{Float32,2}(E) == Eye{Float32}(5,5)

    @test Eye{T}(ones(T, 5, 5)) == E
    @test Eye(ones(T, 5, 5)) == E
end



## Other matrix types
@test Diagonal(Zeros(5)) == Diagonal(zeros(5))
@test_throws MethodError convert(Diagonal, Zeros(5))

@test Diagonal(Zeros(8,5)) == Diagonal(zeros(5))
@test convert(Diagonal, Zeros(5,5)) == Diagonal(zeros(5))
@test_throws BoundsError convert(Diagonal, Zeros(8,5))

@test convert(Diagonal{Int}, Zeros(5,5)) == Diagonal(zeros(Int,5))
@test_throws BoundsError convert(Diagonal{Int}, Zeros(8,5))


@test Diagonal(Eye(8,5)) == Diagonal(ones(5))
@test convert(Diagonal, Eye(5)) == Diagonal(ones(5))
@test_throws BoundsError convert(Diagonal, Eye(8,5))

@test convert(Diagonal{Int}, Eye(5)) == Diagonal(ones(Int,5))
@test_throws BoundsError convert(Diagonal{Int}, Eye(8,5))



@test SparseVector(Zeros(5)) ==
        SparseVector{Float64}(Zeros(5)) ==
        SparseVector{Float64,Int}(Zeros(5)) ==
        AbstractSparseArray(Zeros(5)) ==
        AbstractSparseVector(Zeros(5)) ==
        AbstractSparseArray{Float64}(Zeros(5)) ==
        AbstractSparseVector{Float64}(Zeros(5)) ==
        AbstractSparseVector{Float64,Int}(Zeros(5)) ==
        spzeros(5)

for (Mat, SMat) in ((Zeros(5,5), spzeros(5,5)), (Zeros(6,5), spzeros(6,5)),
                    (Eye(5), sparse(I,5,5)), (Eye(6,5), sparse(I,6,5)))
    @test SparseMatrixCSC(Mat) ==
            SparseMatrixCSC{Float64}(Mat) ==
            SparseMatrixCSC{Float64,Int}(Mat) ==
            AbstractSparseArray(Mat) ==
            AbstractSparseMatrix(Mat) ==
            AbstractSparseArray{Float64}(Mat) ==
            AbstractSparseArray{Float64,Int}(Mat) ==
            AbstractSparseMatrix{Float64}(Mat) ==
            AbstractSparseMatrix{Float64,Int}(Mat) ==
            SMat
end
