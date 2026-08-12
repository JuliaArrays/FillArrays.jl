module FillArraysArrayInterfaceExt

using ArrayInterface

import FillArrays: has_mutable_storage

# `can_setindex` answers the same question across the ecosystem, and more types have declared
# themselves there than would ever declare a hook of ours. It is consulted for arrays only, so
# that it stays less specific than the methods `FillArrays` defines itself, which hold whether or
# not this extension is loaded.
has_mutable_storage(a::AbstractArray) = ArrayInterface.can_setindex(typeof(a))

end # module
