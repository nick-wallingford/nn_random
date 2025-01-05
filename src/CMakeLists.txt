add_executable(nn_random main.cpp)
target_include_directories(nn_random PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

option(sanitize "Use sanitizers")

include(ProcessorCount)
ProcessorCount(CPU_COUNT)
message("cpu count: ${CPU_COUNT}")

configure_file(config.h.in config.h)

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
  target_compile_options(nn_random PRIVATE -march=native -Wall -Wextra)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -fallow-store-data-races")
  endif()

  if(sanitize)
    target_compile_options(nn_random PRIVATE -fsanitize=address)
    target_link_libraries(nn_random PRIVATE asan)

    target_compile_options(nn_random PRIVATE -fsanitize=undefined)
    target_link_libraries(nn_random PRIVATE ubsan)
  endif()
endif()
