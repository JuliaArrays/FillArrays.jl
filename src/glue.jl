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
end