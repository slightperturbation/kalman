#####################################################################
#

cmake_minimum_required(VERSION 2.8.9)
include(ExternalProject)
enable_language(C)
enable_language(CXX)

set( CMAKE_MODULE_PATH "./cmake" "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH} )

#####################################################################
# Common settings across all projects.
#
# Prefer static libraries, so prepend common static library suffixes
# See: http://stackoverflow.com/questions/3762057/cmake-how-to-produce-binaries-as-static-as-possible
IF(WIN32)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
ELSE(WIN32)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
ENDIF(WIN32)

# Use C++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX1Y)
if(COMPILER_SUPPORTS_CXX1Y)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
else(COMPILER_SUPPORTS_CXX1Y)
    CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
    if(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else(COMPILER_SUPPORTS_CXX11)
        CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
        if(COMPILER_SUPPORTS_CXX0X)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        else(COMPILER_SUPPORTS_CXX0X)
            message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Use clang.")
        endif(COMPILER_SUPPORTS_CXX0X)
    endif(COMPILER_SUPPORTS_CXX11)
endif(COMPILER_SUPPORTS_CXX1Y)
if( APPLE )
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++1y" )
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
endif( APPLE )

# Bug in some versions of visual studio creates duplicate manifest entries
# Avoiding manifests entirely seems to be one fix. (http://www.cmake.org/Bug/print_bug_page.php?bug_id=8386)
if( WINDOWS )
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /MANIFEST:NO")
  if( MSVC )
    # warning C4530: C++ exception handler used, but unwind semantics are not 
    # enabled. Specify /EHsc 
    # MSVC needs /EHsc flag, which I included in root CMakeLists.txt, but 
    # above SET(CMAKE_CXX_FLAGS "") erases it.
    SET(CMAKE_CXX_FLAGS "/EHsc")  
  endif( MSVC )
endif( WINDOWS )

#####################################################################
# Find project cmake files
set(CMAKE_MODULE_PATH 
  "${CMAKE_SOURCE_DIR}/cmake/" 
  ${CMAKE_MODULE_PATH}
)

# Include all included external dependency headers
include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/ext/include" )

#####################################################################

## Build Unit Tests
add_subdirectory( test )

## Build Doxygen Docs
add_subdirectory( doxygen )
