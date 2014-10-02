##############################################################################
# Google Test
##############################################################################
include( ExternalProject )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set default ExternalProject root directory
set_directory_properties( PROPERTIES EP_PREFIX "${CMAKE_BINARY_DIR}/ext" )

# from http://stackoverflow.com/questions/9689183/cmake-googletest
ExternalProject_Add(
    googletest
    SVN_REPOSITORY "http://googletest.googlecode.com/svn/trunk/"
    SVN_REVISION -r 660
    TIMEOUT 10
    # Force separate output paths for debug and release builds to allow easy
    # identification of correct lib in subsequent TARGET_LINK_LIBRARIES commands
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
               -Dgtest_force_shared_crt=ON
    # Disable install step
    INSTALL_COMMAND ""
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
)
# Specify include dir
ExternalProject_Get_Property( googletest source_dir )
include_directories( "${source_dir}/include" )

#####################################################################
# Eigen
#####################################################################
INCLUDE(FindPkgConfig )
pkg_check_modules( EIGEN REQUIRED eigen3 )
find_path( EIGEN_INCLUDE_DIR Eigen/Dense
          HINTS ${PC_EIGEN_INCLUDEDIR} ${PC_EIGEN_INCLUDE_DIRS} /usr/local/include/eigen3 )
include_directories( /usr/local/include/eigen3 )

#####################################################################
# Unit Test executable
#####################################################################

# Library includes
include_directories( "${CMAKE_SOURCE_DIR}/include" )

# Add test exe target
add_executable( unitTests ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp )

# Create dependency of unitTests on googletest, forcing build
add_dependencies( unitTests googletest )

# Link with GTest
if( MSVC )
  set(Suffix ".lib")
else()
  set(Suffix ".a")
  set(Pthread "-pthread")
endif()
ExternalProject_Get_Property( googletest binary_dir )
target_link_libraries(
    unitTests
    debug ${binary_dir}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${Suffix}
    optimized ${binary_dir}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${Suffix}
    ${Pthread}
)
