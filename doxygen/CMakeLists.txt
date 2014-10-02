#####################################################################
# Doxygen
#####################################################################
# Add an Option to toggle the generation of the API documentation
OPTION( BUILD_DOCUMENTATION "Use Doxygen to create the HTML based API documentation" ON)
if( BUILD_DOCUMENTATION )
  FIND_PACKAGE( Doxygen )
  if( NOT DOXYGEN_FOUND )
    message( FATAL_ERROR "Doxygen is needed to build the documentation. Install from doxygen.org" )
  endif()

  #-- Configure the Template Doxyfile for our specific project
  set( PROJECT_VERSION "0.1" )
  set( DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/doxygen/")
  set( README_FILE "Readme.md")
  set( SOURCE_DIR "${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/Readme.md" )
  configure_file( "${CMAKE_SOURCE_DIR}/doxygen/Doxyfile.in" 
                  "${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile"  @ONLY IMMEDIATE)
  #-- Add a custom target to run Doxygen when ever the project is built
  add_custom_target( doxygen_docs 
                      COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile"
                      SOURCES ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile )
endif()

