"""Work around ROCm 10 rocPRIM placement-new in HIP device compilation."""

import os
from pathlib import Path


header = (
    Path(os.environ["ROCM_PATH"])
    / "include/rocprim/types/uninitialized_array.hpp"
)
original = "        return *new(ptr) T(std::forward<Args>(args)...);"
replacement = """#if defined(__HIP_DEVICE_COMPILE__)
        static_assert(std::is_trivially_copyable_v<T>);
        *ptr = T(std::forward<Args>(args)...);
        return *ptr;
#else
        return *new(ptr) T(std::forward<Args>(args)...);
#endif"""

source = header.read_text()
if replacement in source:
    print("rocPRIM HIP workaround already applied")
elif source.count(original) == 1:
    header.write_text(source.replace(original, replacement))
    print("Applied rocPRIM HIP workaround")
else:
    raise SystemExit(f"Unexpected rocPRIM header layout: {header}")
