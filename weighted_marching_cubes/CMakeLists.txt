CMAKE_MINIMUM_REQUIRED(VERSION 3.4)

SET(PROJECT_NAME weight_marching_cubes)
SET(CMAKE_BUILD_TYPE RelWithDebInfo)

# Eigen
SET(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen/")
# FIND_PACKAGE( Eigen REQUIRED )
MESSAGE(STATUS "Found Eigen headers in: ${EIGEN_INCLUDE_DIR}")
INCLUDE_DIRECTORIES( ${EIGEN_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mesh/)

# marching cubes
SET( LIB_NAME mc_lib )
ADD_LIBRARY( ${LIB_NAME}
	mesh/MarchingCubes.cpp
	mesh/MarchingCubesNoColor.cpp
    )
TARGET_COMPILE_OPTIONS( ${LIB_NAME} PRIVATE -std=c++17 )

set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC" )

# Find and include Pybind11
add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} src/pywrapper.cpp)
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} PRIVATE
    mc_lib
)
