# QtDeployment.cmake - Bundle Qt redistributables for Windows/macOS packages
#
# Provides blackchirp_deploy_qt(<target>) which registers an install(CODE)
# hook that runs windeployqt (Windows) or macdeployqt (macOS) against the
# installed binary, so the resulting CPack package is self-contained.
#
# On Linux this is a no-op: system package managers resolve Qt via the
# auto-derived shlibs/RPM AUTOREQ machinery configured in Packaging.cmake.
#
# Usage: call blackchirp_deploy_qt(<target>) AFTER the install(TARGETS ...)
# rule for that target. Install rules execute in registration order, so the
# binary must be staged before the deploy tool runs against it.

if(BLACKCHIRP_QT_DEPLOYMENT_INCLUDED)
    return()
endif()
set(BLACKCHIRP_QT_DEPLOYMENT_INCLUDED TRUE)

if(WIN32 OR APPLE)
    if(NOT TARGET Qt6::qmake)
        message(FATAL_ERROR
            "Qt6::qmake target not found; cannot locate Qt deployment tool. "
            "Ensure find_package(Qt6) succeeded before including QtDeployment.cmake.")
    endif()
    get_target_property(_qt_qmake_executable Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(_qt_bin_dir "${_qt_qmake_executable}" DIRECTORY)
endif()

if(WIN32)
    find_program(BLACKCHIRP_WINDEPLOYQT_EXECUTABLE
        NAMES windeployqt windeployqt.exe
        HINTS "${_qt_bin_dir}"
        REQUIRED
    )
    message(STATUS "Found windeployqt: ${BLACKCHIRP_WINDEPLOYQT_EXECUTABLE}")
elseif(APPLE)
    find_program(BLACKCHIRP_MACDEPLOYQT_EXECUTABLE
        NAMES macdeployqt
        HINTS "${_qt_bin_dir}"
        REQUIRED
    )
    message(STATUS "Found macdeployqt: ${BLACKCHIRP_MACDEPLOYQT_EXECUTABLE}")
endif()

# blackchirp_deploy_qt(<target>)
#
# Register an install(CODE) hook that runs the platform Qt deployment tool
# against <target>'s installed binary. Looks up the binary under
# CMAKE_INSTALL_PREFIX at install time so it works for both `cmake --install`
# and `cpack` (which sets CMAKE_INSTALL_PREFIX to a staging directory).
function(blackchirp_deploy_qt target)
    if(WIN32)
        # Locate qwt.dll so windeployqt walks its Qt-module imports as
        # well as the .exe's. Without this, qwt's runtime deps on
        # Qt6OpenGL.dll / Qt6OpenGLWidgets.dll are missed (the .exe
        # itself only links Qt6Widgets / Core / Gui), and the installed
        # package fails with STATUS_DLL_NOT_FOUND. windeployqt accepts
        # multiple input binaries and unions their dep closures.
        set(_qwt_dll_arg)
        if(QWT_LIBRARY)
            get_filename_component(_qwt_lib_dir "${QWT_LIBRARY}" DIRECTORY)
            foreach(_candidate "${_qwt_lib_dir}/qwt.dll"
                               "${_qwt_lib_dir}/../bin/qwt.dll")
                if(EXISTS "${_candidate}")
                    get_filename_component(_qwt_dll "${_candidate}" ABSOLUTE)
                    set(_qwt_dll_arg "${_qwt_dll}")
                    break()
                endif()
            endforeach()
            unset(_qwt_lib_dir)
        endif()

        # --no-compiler-runtime: windeployqt's --compiler-runtime mode
        # ships vc_redist.x64.exe (25 MB installer) rather than the
        # individual DLLs. The build runner has the VC++ runtime in
        # System32 from VS Build Tools, so the smoke test doesn't need
        # them bundled; end-user distribution of the VC++ Redistributable
        # is a separate decision (NSIS-driven auto-run, or document the
        # prerequisite) tracked outside this hook.
        install(CODE "
            set(_exe \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/${target}.exe\")
            if(NOT EXISTS \"\${_exe}\")
                message(FATAL_ERROR \"windeployqt: ${target}.exe not found at \${_exe}\")
            endif()
            set(_extra_targets)
            if(\"${_qwt_dll_arg}\" AND EXISTS \"${_qwt_dll_arg}\")
                list(APPEND _extra_targets \"${_qwt_dll_arg}\")
            endif()
            message(STATUS \"Running windeployqt on \${_exe} \${_extra_targets}\")
            execute_process(
                COMMAND \"${BLACKCHIRP_WINDEPLOYQT_EXECUTABLE}\"
                    --no-translations
                    --no-compiler-runtime
                    --dir \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}\"
                    --verbose 1
                    \"\${_exe}\"
                    \${_extra_targets}
                COMMAND_ERROR_IS_FATAL ANY
            )
        " COMPONENT Applications)
    elseif(APPLE)
        # macdeployqt resolves the binary's load-command closure with otool.
        # Qwt's qmake build on macOS produces a libqwt.6.dylib whose recorded
        # install_name does not point at the from-source install dir, so the
        # binary's LC_LOAD_DYLIB references a file macdeployqt cannot find on
        # the build host. Without `-libpath`, macdeployqt prints
        # `ERROR: no file at ...libqwt.6.dylib` and silently leaves the dylib
        # out of the bundle, producing a .app that fails to launch on a clean
        # Mac. Passing the directory containing the from-source libqwt lets
        # macdeployqt locate the library by basename, copy it into
        # Contents/Frameworks/, and rewrite the load command to point inside
        # the bundle.
        set(_macdeployqt_libpath_args)
        if(QWT_LIBRARY)
            get_filename_component(_qwt_lib_dir "${QWT_LIBRARY}" DIRECTORY)
            list(APPEND _macdeployqt_libpath_args "-libpath=${_qwt_lib_dir}")
        endif()

        # Both apps install with BUNDLE DESTINATION `.` so the .app sits at
        # the install-prefix root (matches DragNDrop DMG layout).
        install(CODE "
            set(_bundle \"\${CMAKE_INSTALL_PREFIX}/${target}.app\")
            if(NOT IS_DIRECTORY \"\${_bundle}\")
                message(FATAL_ERROR \"macdeployqt: ${target}.app not found at \${_bundle}\")
            endif()
            message(STATUS \"Running macdeployqt on \${_bundle}\")
            execute_process(
                COMMAND \"${BLACKCHIRP_MACDEPLOYQT_EXECUTABLE}\"
                    \"\${_bundle}\"
                    ${_macdeployqt_libpath_args}
                    -verbose=1
                COMMAND_ERROR_IS_FATAL ANY
            )
        " COMPONENT Applications)
    endif()
endfunction()
