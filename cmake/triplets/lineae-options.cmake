# The pinned ONNX Runtime port explicitly requires this ONNX build option.
# Apply it to the dependency, not to the consuming application's CMake cache.
if(PORT STREQUAL "onnx")
  list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DONNX_DISABLE_STATIC_REGISTRATION=ON)
endif()
