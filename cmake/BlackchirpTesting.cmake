# BlackchirpTesting.cmake - Helpers for declaring test targets
#
# Provides blackchirp_add_gui_test(), which declares a Qt-Test executable
# that links the whole GUI stack. A GUI test needs the acquisition and
# hardware libraries as well as blackchirp-gui itself: AUTOMOC emits one
# mocs_compilation.cpp per target, so pulling the moc for any single
# Q_OBJECT GUI class drags in MainWindow, and with it AcquisitionManager
# and HardwareManager.
#
# Resources are deliberately not compiled into GUI tests. GUI classes
# request icons through ThemeColors::createThemedIcon(), which returns a
# null QIcon when the resource cannot be opened; nothing downstream
# treats that as an error.

# Include guard to prevent multiple inclusions
if(BLACKCHIRP_TESTING_CMAKE_INCLUDED)
    return()
endif()
set(BLACKCHIRP_TESTING_CMAKE_INCLUDED TRUE)

# ============================================================================
# blackchirp_add_gui_test
# ============================================================================
#
#   blackchirp_add_gui_test(<target>
#       NAME         <ctest name>
#       [SOURCES     <file> ...]     # defaults to tests/<target>.cpp
#       [LIBRARIES   <lib> ...]      # extra libraries beyond the GUI stack
#       [DEFINITIONS <def> ...]      # extra PRIVATE compile definitions
#       [TESTDATA]                   # define TESTDATA_DIR
#   )
#
# Registers the executable with CTest under an offscreen QPA platform, and
# adds it to the list returned by blackchirp_get_gui_tests() so the `tests`
# aggregate target can depend on it.
function(blackchirp_add_gui_test target)
    cmake_parse_arguments(BCGT
        "TESTDATA"
        "NAME"
        "SOURCES;LIBRARIES;DEFINITIONS"
        ${ARGN}
    )

    if(NOT BCGT_NAME)
        message(FATAL_ERROR
            "blackchirp_add_gui_test(${target}): NAME is required")
    endif()

    if(NOT BCGT_SOURCES)
        set(BCGT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tests/${target}.cpp)
    endif()

    add_executable(${target} ${BCGT_SOURCES})

    # blackchirp-gui exports Qt6::Widgets, QWT and the data layer through
    # its PUBLIC link interface; they are named anyway so a future change
    # to that interface surfaces as a decision rather than as a broken
    # test link.
    target_link_libraries(${target}
        blackchirp-gui
        blackchirp-acquisition
        blackchirp-hardware
        blackchirp-data
        Qt6::Test
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::Concurrent
        QWT::QWT
        ${BCGT_LIBRARIES}
    )

    add_blackchirp_definitions(${target})

    if(BCGT_TESTDATA)
        list(APPEND BCGT_DEFINITIONS
            TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/testdata")
    endif()

    if(BCGT_DEFINITIONS)
        target_compile_definitions(${target} PRIVATE ${BCGT_DEFINITIONS})
    endif()

    add_test(NAME ${BCGT_NAME} COMMAND ${target})
    set_tests_properties(${BCGT_NAME} PROPERTIES
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
    )

    set_property(GLOBAL APPEND PROPERTY BLACKCHIRP_GUI_TEST_TARGETS ${target})
endfunction()

# Returns every target declared by blackchirp_add_gui_test() so far.
function(blackchirp_get_gui_tests out_var)
    get_property(_targets GLOBAL PROPERTY BLACKCHIRP_GUI_TEST_TARGETS)
    set(${out_var} ${_targets} PARENT_SCOPE)
endfunction()
