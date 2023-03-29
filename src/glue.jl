using Requires

function __init__()
    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
        @_add_fill StaticArrays.StaticArray
    end
    @require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" begin
        @_add_fill LazyArrays.Vcat
    end
end