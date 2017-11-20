using FillArrays, Base.Test


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
    @test eltype(E) == T
    @test Array(E) == eye(T, 5, 5)
    @test Array{T}(E) == eye(T, 5, 5)
    @test Array{T,2}(E) == eye(T, 5, 5)

    @test AbstractArray(E) === E
    @test AbstractArray{T}(E) === E
    @test AbstractArray{T,2}(E) === E


    @test AbstractArray{Float32}(E) == Eye{Float32}(5,5)
    @test AbstractArray{Float32,2}(E) == Eye{Float32}(5,5)

    @test Eye{T}(ones(T, 5, 5)) == E
    @test Eye(ones(T, 5, 5)) == E
end
