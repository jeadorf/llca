# This file might work with older versions, but it has only been tested with
# CMake versions >= 2.8.7.
cmake_minimum_required(VERSION 2.8.10)

project(llca)

# =============================================================================
# BUILD OPTIONS
# =============================================================================

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "-g -O2 -fPIC -fopenmp -pedantic -Wall -Wextra -Wunreachable-code")

# =============================================================================
# SOURCES AND HEADERS
# =============================================================================

file(GLOB srcs src/*.cpp)
include_directories(include)

# -----------------------------------------------------------------------------
# BUILD MAGIC
# -----------------------------------------------------------------------------

include_directories("${PROJECT_BINARY_DIR}/include")
configure_file(
    "${PROJECT_SOURCE_DIR}/include/llca/configure.hpp.in"
    "${PROJECT_BINARY_DIR}/include/llca/configure.hpp")

# =============================================================================
# DEPENDENCY: GSL
# =============================================================================

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
link_directories(${GSL_LINK_DIRS})

# =============================================================================
# DEPENDENCY: GOMP
# =============================================================================

set(GOMP_LIBRARIES gomp)

# =============================================================================
# DEPENDENCY: BOOST
# =============================================================================

find_package(Boost REQUIRED COMPONENTS filesystem program_options system
    unit_test_framework)
include_directories(${Boost_INCLUDE_DIRS})

# =============================================================================
# DEPENDENCY: KMLOCAL
# =============================================================================

if (DEFINED KMLOCAL_SOURCE_DIR)
    include_directories(${KMLOCAL_SOURCE_DIR}/src)
    file(GLOB kmlocal_srcs ${KMLOCAL_SOURCE_DIR}/src/*.cpp)
    list(REMOVE_ITEM kmlocal_srcs ${KMLOCAL_SOURCE_DIR}/src/kmltest.cpp)
    list(REMOVE_ITEM kmlocal_srcs ${KMLOCAL_SOURCE_DIR}/src/kmlsample.cpp)
    list(REMOVE_ITEM kmlocal_srcs ${KMLOCAL_SOURCE_DIR}/src/kmlminimal.cpp)
    add_library(kmlocal SHARED ${kmlocal_srcs})
    set(KMLOCAL_LIBRARIES kmlocal)
else()
    message(FATAL_ERROR "You need to set KMLOCAL_SOURCE_DIR. "
                    "This is the folder where the kmlocal sources resides, "
                    "for example /path/to/kmlocal-1.7.2. You can pass this to "
                    "CMake via the -D option. See the CMake documentation for "
                    "further help.")
endif()

# =============================================================================
# DEPENDENCY: LIBSVM
# =============================================================================

if (DEFINED LIBSVM_SOURCE_DIR)
    include_directories(${LIBSVM_SOURCE_DIR})
    file(GLOB svm_srcs ${LIBSVM_SOURCE_DIR}/*.cpp)
    add_library(svm SHARED ${svm_srcs})
    set(LIBSVM_LIBRARIES svm)
else()
    message(FATAL_ERROR "You need to set LIBSVM_SOURCE_DIR. "
                    "This is the folder where the libsvm sources resides, "
                    "for example /path/to/libsvm-3.17. You can pass this to "
                    "CMake via the -D option. See the CMake documentation for "
                    "further help.")
endif()

# =============================================================================
# LIBRARIES
# =============================================================================

add_library(${PROJECT_NAME} SHARED ${srcs})
target_link_libraries(${PROJECT_NAME}
                        ${GSL_LIBRARIES}
                        ${Boost_LIBRARIES}
                        ${GOMP_LIBRARIES}
                        ${KMLOCAL_LIBRARIES}
                        ${LIBSVM_LIBRARIES})

# =============================================================================
# TESTING
# =============================================================================

enable_testing()

file(COPY "${PROJECT_SOURCE_DIR}/data" DESTINATION "${PROJECT_BINARY_DIR}")

file(GLOB tests test/*.cpp)
foreach(t ${tests})
    get_filename_component(n ${t} NAME_WE)
    add_executable(${n} ${t})
    target_link_libraries(${n} ${PROJECT_NAME})
    add_test(${n} ${n})
endforeach(t)
