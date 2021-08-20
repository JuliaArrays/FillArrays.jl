import ChainRulesCore: ProjectTo, NoTangent, Tangent

"""
    ProjectTo(::Fill) -> ProjectTo{Fill}
    ProjectTo(::Ones) -> ProjectTo{NoTangent}

Most FillArrays arrays store one number, and so their gradients under automatic
differentiation represent the variation of this one number. 

The exception is those like `Ones` and `Zeros` whose type fixes their value,
which have no graidient.
"""
ProjectTo(x::Fill{<:Number}) = ProjectTo{Fill}(; element = ProjectTo(getindex_value(x)), axes = axes(x))

ProjectTo(x::AbstractFill{Bool}) = ProjectTo{NoTangent}()  # Bool is always regarded as categorical

ProjectTo(x::Zeros) = ProjectTo{NoTangent}()
ProjectTo(x::Ones) = ProjectTo{NoTangent}()

function (project::ProjectTo{Fill})(dx::AbstractArray)
    for d in 1:max(ndims(dx), length(project.axes))
        size(dx, d) == length(get(project.axes, d, 1)) || throw(_projection_mismatch(axes_x, size(dx)))
    end
    Fill(mean(dx), project.axes)  # Note that mean(dx::Fill) is optimised
end

function (project::ProjectTo{Fill})(dx::Tangent{<:Fill})
    # This would need a definition for length(::NoTangent) to be safe:
    # for d in 1:max(length(dx.axes), length(project.axes))
    #     length(get(dx.axes, d, 1)) == length(get(project.axes, d, 1)) || throw(_projection_mismatch(dx.axes, size(dx)))
    # end
    Fill(dx.value / prod(length, project.axes), project.axes)
end

function _projection_mismatch(axes_x::Tuple, size_dx::Tuple)
    size_x = map(length, axes_x)
    DimensionMismatch("variable with size(x) == $size_x cannot have a gradient with size(dx) == $size_dx")
end
