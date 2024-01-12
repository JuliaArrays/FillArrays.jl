using Aqua
using FillArrays
using Test
@testset "Project quality" begin
    Aqua.test_all(FillArrays;
        # https://github.com/JuliaArrays/FillArrays.jl/issues/105#issuecomment-1582516319
        ambiguities=(; broken=true),
    )
end
