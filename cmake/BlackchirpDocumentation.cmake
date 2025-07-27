# BlackchirpDocumentation.cmake - Documentation build targets
#
# This module defines documentation build targets for Blackchirp using Sphinx 
# and Doxygen. It creates targets for building HTML documentation, API docs,
# and integrated documentation with C++ API references.

# Include guard to prevent multiple inclusions
if(BLACKCHIRP_DOCUMENTATION_CMAKE_INCLUDED)
    return()
endif()
set(BLACKCHIRP_DOCUMENTATION_CMAKE_INCLUDED TRUE)

# ============================================================================
# Find Required Documentation Tools
# ============================================================================

# Find Sphinx for main documentation
find_program(SPHINX_BUILD_EXECUTABLE
    NAMES sphinx-build sphinx-build.exe
    DOC "Path to sphinx-build executable"
)

# Find Doxygen for API documentation
find_package(Doxygen QUIET COMPONENTS dot)

# ============================================================================
# Documentation Configuration
# ============================================================================

# Documentation directories
set(SPHINX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/doc/source")
set(SPHINX_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs")
set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen")

# Sphinx build options
set(SPHINX_OPTS "" CACHE STRING "Additional options to pass to sphinx-build")

# ============================================================================
# Doxygen API Documentation Target
# ============================================================================

if(DOXYGEN_FOUND)
    # Configure Doxygen
    set(DOXYGEN_INPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
    set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen")
    set(DOXYGEN_INDEX_FILE "${DOXYGEN_OUTPUT_DIR}/xml/index.xml")
    
    # Configure Doxyfile from template
    if(EXISTS "${SPHINX_SOURCE_DIR}/Doxyfile.in")
        # Use the Doxyfile.in template with CMake variable substitution
        configure_file(
            "${SPHINX_SOURCE_DIR}/Doxyfile.in"
            "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
            @ONLY
        )
    else()
        # Fallback: create a basic Doxyfile if template doesn't exist
        message(WARNING "Doxyfile.in not found, creating minimal Doxyfile")
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Doxyfile.in"
            "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
            @ONLY
        )
    endif()
    
    # Create Doxygen target
    add_custom_command(
        OUTPUT ${DOXYGEN_INDEX_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${DOXYGEN_OUTPUT_DIR}
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        MAIN_DEPENDENCY ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
    
    # Create doxygen target
    add_custom_target(doxygen
        DEPENDS ${DOXYGEN_INDEX_FILE}
    )
    
    message(STATUS "Doxygen found: API documentation target 'doxygen' available")
else()
    message(STATUS "Doxygen not found: API documentation will not be available")
endif()

# ============================================================================
# Sphinx Documentation Targets
# ============================================================================

if(SPHINX_BUILD_EXECUTABLE)
    # Ensure build directory exists
    file(MAKE_DIRECTORY ${SPHINX_BUILD_DIR})
    
    # Find all source files for dependency tracking
    file(GLOB_RECURSE SPHINX_SOURCES 
        "${SPHINX_SOURCE_DIR}/*.rst"
        "${SPHINX_SOURCE_DIR}/*.md"
        "${SPHINX_SOURCE_DIR}/*.py"
        "${SPHINX_SOURCE_DIR}/*.txt"
    )
    
    # HTML documentation target
    add_custom_command(
        OUTPUT "${SPHINX_BUILD_DIR}/html/index.html"
        COMMAND ${SPHINX_BUILD_EXECUTABLE}
            -M html
            ${SPHINX_SOURCE_DIR}
            ${SPHINX_BUILD_DIR}
            ${SPHINX_OPTS}
        DEPENDS ${SPHINX_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc
        COMMENT "Building HTML documentation with Sphinx"
        VERBATIM
    )
    
    # Create main documentation target
    add_custom_target(docs
        DEPENDS "${SPHINX_BUILD_DIR}/html/index.html"
    )
    
    # Add doxygen dependency if available
    if(DOXYGEN_FOUND)
        add_dependencies(docs doxygen)
    endif()
    
    # PDF documentation target (if LaTeX is available)
    find_program(LATEX_EXECUTABLE NAMES pdflatex)
    if(LATEX_EXECUTABLE)
        add_custom_target(docs-pdf
            COMMAND ${SPHINX_BUILD_EXECUTABLE}
                -M latexpdf
                ${SPHINX_SOURCE_DIR}
                ${SPHINX_BUILD_DIR}
                ${SPHINX_OPTS}
            DEPENDS ${SPHINX_SOURCES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc
            COMMENT "Building PDF documentation with Sphinx"
            VERBATIM
        )
        
        if(DOXYGEN_FOUND)
            add_dependencies(docs-pdf doxygen)
        endif()
        
        message(STATUS "LaTeX found: PDF documentation target 'docs-pdf' available")
    endif()
    
    # Documentation testing target
    add_custom_target(docs-linkcheck
        COMMAND ${SPHINX_BUILD_EXECUTABLE}
            -M linkcheck
            ${SPHINX_SOURCE_DIR}
            ${SPHINX_BUILD_DIR}
            ${SPHINX_OPTS}
        DEPENDS ${SPHINX_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc
        COMMENT "Checking documentation links with Sphinx"
        VERBATIM
    )
    
    # Clean documentation target
    add_custom_target(docs-clean
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${SPHINX_BUILD_DIR}
        COMMENT "Cleaning documentation build directory"
    )
    
    message(STATUS "Sphinx found: Documentation targets available:")
    message(STATUS "  docs          - Build HTML documentation")
    message(STATUS "  docs-linkcheck - Check documentation links")
    message(STATUS "  docs-clean    - Clean documentation build")
    if(LATEX_EXECUTABLE)
        message(STATUS "  docs-pdf      - Build PDF documentation")
    endif()
    
else()
    message(STATUS "Sphinx not found: Documentation targets will not be available")
    message(STATUS "  Install Sphinx to enable documentation building:")
    message(STATUS "    pip install sphinx sphinx_rtd_theme breathe")
endif()

# ============================================================================
# Installation Configuration
# ============================================================================

if(SPHINX_BUILD_EXECUTABLE)
    # Install HTML documentation
    install(
        DIRECTORY "${SPHINX_BUILD_DIR}/html/"
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/html
        COMPONENT Documentation
        OPTIONAL
    )
    
    # Install API documentation if available
    if(DOXYGEN_FOUND)
        install(
            DIRECTORY "${DOXYGEN_OUTPUT_DIR}/html/"
            DESTINATION ${CMAKE_INSTALL_DOCDIR}/api
            COMPONENT Documentation
            OPTIONAL
        )
    endif()
endif()

# ============================================================================
# Status Information
# ============================================================================

message(STATUS "Blackchirp Documentation Configuration:")
message(STATUS "  Sphinx executable: ${SPHINX_BUILD_EXECUTABLE}")
message(STATUS "  Doxygen found: ${DOXYGEN_FOUND}")
message(STATUS "  Source directory: ${SPHINX_SOURCE_DIR}")
message(STATUS "  Build directory: ${SPHINX_BUILD_DIR}")

if(SPHINX_BUILD_EXECUTABLE)
    message(STATUS "  Documentation will be built to: ${SPHINX_BUILD_DIR}/html/")
    message(STATUS "  Use 'make docs' to build documentation")
endif()