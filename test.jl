using FillArrays, BenchmarkTools
import Base: *
using Test

# const FillVector{F,A} = Fill{F,1,A}
# const FillMatrix{F,A} = Fill{F,2,A}
# const OnesVector{F,A} = Ones{F,1,A}
# const OnesMatrix{F,A} = Ones{F,2,A}
# const ZerosVector{F,A} = Zeros{F,1,A}
# const ZerosMatrix{F,A} = Zeros{F,2,A}

# function *(x::AbstractMatrix, f::FillMatrix)
#     axes(x, 2) ≠ axes(f, 1) &&
#         throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
#     m = size(f, 2)
#     repeat(sum(x, dims=2) * f.value, 1, m) 
# end

# function *(f::FillMatrix, x::AbstractMatrix)
#     axes(f, 2) ≠ axes(x, 1) &&
#         throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
#     m = size(f, 1)
#     repeat(sum(x, dims=1) * f.value, m, 1) 
# end

# function *(x::AbstractMatrix, f::OnesMatrix)
#     axes(x, 2) ≠ axes(f, 1) &&
#         throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
#     m = size(f, 2)
#     repeat(sum(x, dims=2) * one(eltype(f)), 1, m) 
# end

# function *(f::OnesMatrix, x::AbstractMatrix)
#     axes(f, 2) ≠ axes(x, 1) &&
#         throw(DimensionMismatch("Incompatible matrix multiplication dimensions"))
#     m = size(f, 1)
#     repeat(sum(x, dims=1) * one(eltype(f)), m, 1) 
# end

# function *(a::Adjoint{T, <:StridedMatrix{T}}, b::Fill{T, 2}) where T
#     fB = similar(parent(a), size(b, 1), size(b, 2))
#     fill!(fB, b.value)
#     return a*fB
# end

# function *(a::Transpose{T, <:StridedMatrix{T}}, b::Fill{T, 2}) where T
#     fB = similar(parent(a), size(b, 1), size(b, 2))
#     fill!(fB, b.value)
#     return a*fB
# end

# function *(a::StridedMatrix{T}, b::Fill{T, 2}) where T
#     fB = similar(a, size(b, 1), size(b, 2))
#     fill!(fB, b.value)
#     return a*fB
# end

# @test x * o ≈ x * ones(n,n)
# @test o * x ≈ ones(n,n) * x

for n in [2, 5, 10, 20, 200, 2000]
    x, y = rand(n, n), rand(n, n)
    o, f = Ones(n, n), Fill(1.0, n, n)

    println("RIGHT n=$n")
    @btime $x * $y
    @btime $x * $o
    @btime $x * $f

    println("LEFT n=$n")
    @btime $x * $y
    @btime $o * $y
    @btime $f * $y

    println()
end