
function Base.cat_t(::Type{T}, fs::Fill...; dims) where T
    allvals = unique([f.value for f in fs])
    length(allvals) > 1 && return Base._cat_t(dims, T,  fs...)

    catdims = Base.dims2cat(dims)

    # Note, when dims is a tuple the output gets zero padded and we can't use a Fill unless it is all zeros too
    allvals[] !== zero(T) && sum(catdims) > 1 && return Base._cat_t(dims, T, fs...)

    shape = Base.cat_shape(catdims, map(Base.cat_size, fs)::Tuple{Vararg{Union{Int,Dims}}})::Dims
    return Fill(convert(T, fs[1].value), shape)
end

Base.vcat(vs::Fill...) = cat(vs...;dims=Val(1))
Base.hcat(vs::Fill...) = cat(vs...;dims=Val(2))