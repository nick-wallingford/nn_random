add_executable(nn_random main.cpp)

option(sanitize "Use sanitizers")

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
  target_compile_options(nn_random PRIVATE -march=native -Wall -Wextra)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")
  endif()

  if(sanitize)
    target_compile_options(nn_random PRIVATE -fsanitize=address)
    target_link_libraries(nn_random PRIVATE asan)

    target_compile_options(nn_random PRIVATE -fsanitize=undefined)
    target_link_libraries(nn_random PRIVATE ubsan)
  endif()
endif()

