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
        # Locate qwt.dll at configure time. Without baking the resolved
        # path into the install(CODE) string literally, the deferred
        # variable-expansion path (set var in install code, splice into
        # COMMAND args via `\${var}`) silently produced an empty arg
        # list and windeployqt only saw the .exe — see commit history
        # for the prior attempt and the diagnostic that surfaced it.
        set(_qwt_dll "")
        if(QWT_LIBRARY)
            get_filename_component(_qwt_lib_dir "${QWT_LIBRARY}" DIRECTORY)
            foreach(_candidate "${_qwt_lib_dir}/qwt.dll"
                               "${_qwt_lib_dir}/../bin/qwt.dll")
                if(EXISTS "${_candidate}")
                    get_filename_component(_qwt_dll "${_candidate}" ABSOLUTE)
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
            set(_install_bin \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}\")
            message(STATUS \"Running windeployqt on \${_exe}\")
            execute_process(
                COMMAND \"${BLACKCHIRP_WINDEPLOYQT_EXECUTABLE}\"
                    --no-translations
                    --no-compiler-runtime
                    --dir \"\${_install_bin}\"
                    --verbose 1
                    \"\${_exe}\"
                COMMAND_ERROR_IS_FATAL ANY
            )
        " COMPONENT Applications)

        # Second windeployqt pass against qwt.dll, bundling its Qt
        # transitive deps (Qt6OpenGL.dll / Qt6OpenGLWidgets.dll) into
        # the same install bin/. Required because the .exe doesn't
        # directly import those — only qwt.dll does, and windeployqt
        # walks only the binaries it is explicitly given. Two distinct
        # install(CODE) blocks rather than one with conditional args:
        # the conditional-arg form silently dropped the second binary
        # under install(CODE)'s deferred-expansion semantics.
        if(_qwt_dll)
            install(CODE "
                set(_install_bin \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}\")
                message(STATUS \"Running windeployqt on ${_qwt_dll}\")
                execute_process(
                    COMMAND \"${BLACKCHIRP_WINDEPLOYQT_EXECUTABLE}\"
                        --no-translations
                        --no-compiler-runtime
                        --dir \"\${_install_bin}\"
                        --verbose 1
                        \"${_qwt_dll}\"
                    COMMAND_ERROR_IS_FATAL ANY
                )
            " COMPONENT Applications)
        endif()
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
