using Requires

function __init__()
    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
        +(a::StaticArrays.StaticArray, b::Zeros) = abs_add_zero(a, b)
        +(a::Zeros, b::StaticArrays.StaticArray) = abs_add_zero(b, a)
        -(a::StaticArrays.StaticArray, b::Zeros) = abs_add_zero(a, b)
        -(a::Zeros, b::StaticArrays.StaticArray) = abs_add_zero(-b, a)

        +(a::StaticArrays.StaticArray, b::AbstractFill) = fill_add(a, b)
        +(a::AbstractFill, b::StaticArrays.StaticArray) = fill_add(b, a)
        -(a::StaticArrays.StaticArray, b::AbstractFill) = fill_add(a, -b)
        -(a::AbstractFill, b::StaticArrays.StaticArray) = fill_add(-b, a)
    end
    @require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" begin
        +(a::LazyArrays.Vcat, b::Zeros) = abs_add_zero(a, b)
        +(a::Zeros, b::LazyArrays.Vcat) = abs_add_zero(b, a)
        -(a::LazyArrays.Vcat, b::Zeros) = abs_add_zero(a, b)
        -(a::Zeros, b::LazyArrays.Vcat) = abs_add_zero(-b, a)

        +(a::LazyArrays.Vcat, b::AbstractFill) = fill_add(a, b)
        +(a::AbstractFill, b::LazyArrays.Vcat) = fill_add(b, a)
        -(a::LazyArrays.Vcat, b::AbstractFill) = fill_add(a, -b)
        -(a::AbstractFill, b::LazyArrays.Vcat) = fill_add(-b, a)
    end
end