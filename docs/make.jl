using Documenter
using FillArrays

# Setup for doctests in docstrings
DocMeta.setdocmeta!(FillArrays, :DocTestSetup, :(using FillArrays))

makedocs(;
    format = Documenter.HTML(
        canonical = "https://JuliaArrays.github.io/FillArrays.jl/stable/",
    ),
    pages = [
        "Home" => "index.md",
        ],
    sitename = "FillArrays.jl",
)

deploydocs(; repo = "github.com/JuliaArrays/FillArrays.jl")
