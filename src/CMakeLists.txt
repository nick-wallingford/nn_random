add_executable(nn_random main.cpp)

option(sanitize "Use sanitizers")

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
  target_compile_options(nn_random PRIVATE -march=native -funsafe-math-optimizations -Wall -Wextra)

  string(REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fpeel-loops")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fpeel-loops")
  endif()

  if(sanitize)
    target_compile_options(nn_random PRIVATE -fsanitize=address)
    target_link_libraries(nn_random PRIVATE asan)

    target_compile_options(nn_random PRIVATE -fsanitize=undefined)
    target_link_libraries(nn_random PRIVATE ubsan)
  endif()
endif()

