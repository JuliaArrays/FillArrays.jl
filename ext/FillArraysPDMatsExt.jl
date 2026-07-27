module FillArraysPDMatsExt

import FillArrays
import FillArrays.LinearAlgebra
import PDMats
using FillArrays: mult_zeros, AbstractZeros
using PDMats: ScalMat

function PDMats.AbstractPDMat(a::LinearAlgebra.Diagonal{T,<:FillArrays.AbstractFill{T,1}}) where {T<:Real}
    dim = size(a, 1)
    return ScalMat(dim, FillArrays.getindex_value(a.diag))
end

Base.:*(a::ScalMat, b::AbstractZeros{T, 1} where T) = mult_zeros(a, b)
Base.:*(a::ScalMat, b::AbstractZeros{T, 2} where T) = mult_zeros(a, b)
Base.:*(a::AbstractZeros{T, 2} where T, b::ScalMat) = mult_zeros(a, b) # This is implemented in case ScalMat implements right multiplication

end # module
