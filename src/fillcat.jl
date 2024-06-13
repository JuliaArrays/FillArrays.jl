
function Base.cat_t(::Type{T}, fs::Fill...; dims) where T
    allvals = unique([f.value for f in fs])
    length(allvals) > 1 && return Base._cat_t(dims, T,  fs...)

    catdims = Base.dims2cat(dims)

    # When dims is a tuple the output gets zero padded and we can't use a Fill unless it is all zeros
    # There might be some cases when it does not get padded which are not considered here
    
    if sum(catdims) > 1 
        allvals[] isa Number || return Base._cat_t(dims, T, fs...)
        allvals[] !== zero(T) && return Base._cat_t(dims, T, fs...)
    end

    shape = cat_shape_fill(catdims, fs)
    return Fill(convert(T, fs[1].value), shape)
end

Base.vcat(vs::Fill...) = cat(vs...;dims=Val(1))
Base.hcat(vs::Fill...) = cat(vs...;dims=Val(2))


function Base.cat_t(::Type{T}, fs::Zeros...; dims) where T
    catdims = Base.dims2cat(dims)
    shape = cat_shape_fill(catdims, fs)
    return Zeros{T}(shape)
end

Base.vcat(vs::Zeros...) = cat(vs...;dims=Val(1))
Base.hcat(vs::Zeros...) = cat(vs...;dims=Val(2))


function Base.cat_t(::Type{T}, fs::Ones...; dims) where T
    catdims = Base.dims2cat(dims)

    # When dims is a tuple the output gets zero padded so we can't return a Ones
    # There might be some cases when it does not get padded which are not considered here
    sum(catdims) > 1 && return Base._cat_t(dims, T, fs...)

    shape = cat_shape_fill(catdims, fs)
    return Ones{T}(shape)
end

Base.vcat(vs::Ones...) = cat(vs...;dims=Val(1))
Base.hcat(vs::Ones...) = cat(vs...;dims=Val(2))


if VERSION < v"1.6-"
    cat_shape_fill(catdims, fs) = Base.cat_shape(catdims, (), map(Base.cat_size, fs)...)
else
    cat_shape_fill(catdims, fs) = Base.cat_shape(catdims, map(Base.cat_size, fs)::Tuple{Vararg{Union{Int,Dims}}})::Dims
end
