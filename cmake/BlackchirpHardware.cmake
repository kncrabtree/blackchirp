# BlackchirpHardware.cmake - Hardware layer configuration

if(BLACKCHIRP_HARDWARE_CMAKE_INCLUDED)
    return()
endif()
set(BLACKCHIRP_HARDWARE_CMAKE_INCLUDED TRUE)

# Hardware system sources (core infrastructure)
set(HARDWARE_SYSTEM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/hardwaremanager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/hardwareobject.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/hardwareregistry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/hardwareregistration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/runtimehardwareconfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/hardwareprofilemanager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/library/vendorlibrary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/library/spectrumlibrary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/library/labjacklibrary.cpp
)

# Communication protocol sources
file(GLOB COMMUNICATION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/communication/*.cpp
)

# Hardware type base classes
set(HARDWARE_TYPES_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/clock/clockmanager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/clock/clock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/ftmwscope.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/chirpsource/awg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pulsegenerator/pulsegenerator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/flowcontroller/flowcontroller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/gpibcontroller/gpibcontroller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/ioboard/ioboard.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pressurecontroller/pressurecontroller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/tempcontroller/temperaturecontroller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/lifdigitizer/lifscope.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/liflaser/liflaser.cpp
)

# Hardware implementations
file(GLOB HARDWARE_IMPLEMENTATIONS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/virtual*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/dsa*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/m4i*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/dso*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/dpo*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/mso*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/clock/fixedclock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/clock/valon*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/clock/hp*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/chirpsource/virtual*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/chirpsource/awg*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/chirpsource/ad*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/chirpsource/m8*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pulsegenerator/virtual*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pulsegenerator/qc*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pulsegenerator/bnc*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pulsegenerator/srs*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/flowcontroller/virtual*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/flowcontroller/mks*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/gpibcontroller/virtual*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/gpibcontroller/prologix*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/ioboard/virtual*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/ioboard/labjack*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/ioboard/u3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pressurecontroller/virtual*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pressurecontroller/intellisys*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/tempcontroller/virtual*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/tempcontroller/lakeshore*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/lifdigitizer/virtual*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/lifdigitizer/m4i*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/lifdigitizer/rigol*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/liflaser/virtual*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/liflaser/opolette.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/liflaser/sirah*.cpp
)

# Python hardware sources (QProcess-based, no Python build dependencies)
if(BC_ENABLE_PYTHON_HARDWARE)
    file(GLOB PYTHON_HARDWARE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/python/*.cpp
    )
    file(GLOB PYTHON_HARDWARE_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/python/*.h
    )

    # Copy python_hw_host.py to build directory for development
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/python/python_hw_host.py
        ${CMAKE_BINARY_DIR}/python_hw_host.py
        COPYONLY
    )

    # Install python_hw_host.py alongside the application
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/python/python_hw_host.py
            DESTINATION ${CMAKE_INSTALL_DATADIR}/blackchirp)
else()
    set(PYTHON_HARDWARE_SOURCES "")
    set(PYTHON_HARDWARE_HEADERS "")
endif()

# Combine all sources
set(BLACKCHIRP_HARDWARE_SOURCES
    ${HARDWARE_SYSTEM_SOURCES}
    ${COMMUNICATION_SOURCES}
    ${HARDWARE_TYPES_SOURCES}
    ${HARDWARE_IMPLEMENTATIONS_SOURCES}
    ${PYTHON_HARDWARE_SOURCES}
)

# Generate hw_h.h with all hardware headers (types and implementations)
set(HARDWARE_TYPE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/clock/clock.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/ftmwscope.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/chirpsource/awg.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pulsegenerator/pulsegenerator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/flowcontroller/flowcontroller.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/gpibcontroller/gpibcontroller.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/ioboard/ioboard.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pressurecontroller/pressurecontroller.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/tempcontroller/temperaturecontroller.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/lifdigitizer/lifscope.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/liflaser/liflaser.h
)

file(GLOB HARDWARE_IMPLEMENTATION_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/virtual*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/dsa*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/m4i*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/dso*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/dpo*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/mso*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/clock/fixedclock.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/clock/valon*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/clock/hp*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/chirpsource/virtual*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/chirpsource/awg*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/chirpsource/ad*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/chirpsource/m8*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pulsegenerator/virtual*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pulsegenerator/qc*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pulsegenerator/bnc*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pulsegenerator/srs*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/flowcontroller/virtual*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/flowcontroller/mks*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/gpibcontroller/virtual*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/gpibcontroller/prologix*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/ioboard/virtual*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/ioboard/labjack*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pressurecontroller/virtual*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pressurecontroller/intellisys*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/tempcontroller/virtual*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/tempcontroller/lakeshore*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/lifdigitizer/virtual*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/lifdigitizer/m4i*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/lifdigitizer/rigol*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/liflaser/virtual*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/liflaser/opolette.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/liflaser/sirah*.h
)

# Generate hw_base.h content (base types only)
set(HW_BASE_CONTENT "// Generated by CMake - Hardware base types only\n")
foreach(HEADER ${HARDWARE_TYPE_HEADERS})
    file(RELATIVE_PATH REL_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src ${HEADER})
    set(HW_BASE_CONTENT "${HW_BASE_CONTENT}#include \"${REL_HEADER}\"\n")
endforeach()
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/hw_base.h "${HW_BASE_CONTENT}")

# Generate hw_impl.h content (implementations only)  
set(HW_IMPL_CONTENT "// Generated by CMake - Hardware implementations only\n")
foreach(HEADER ${HARDWARE_IMPLEMENTATION_HEADERS})
    file(RELATIVE_PATH REL_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src ${HEADER})
    set(HW_IMPL_CONTENT "${HW_IMPL_CONTENT}#include \"${REL_HEADER}\"\n")
endforeach()
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/hw_impl.h "${HW_IMPL_CONTENT}")

# Generate hw_h.h content (includes both base types and implementations)
set(HW_H_CONTENT "// Generated by CMake - includes both base types and implementations\n")
set(HW_H_CONTENT "${HW_H_CONTENT}#include \"hw_base.h\"\n")
set(HW_H_CONTENT "${HW_H_CONTENT}#include \"hw_impl.h\"\n")
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/hw_h.h "${HW_H_CONTENT}")

# Create hardware library
add_library(blackchirp-hardware STATIC ${BLACKCHIRP_HARDWARE_SOURCES})
add_library(Blackchirp::Hardware ALIAS blackchirp-hardware)

target_include_directories(blackchirp-hardware PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(blackchirp-hardware
    PUBLIC
        Qt6::Core
        Qt6::SerialPort
        Qt6::Network
        blackchirp-data
)

# Add version and configuration definitions
add_blackchirp_definitions(blackchirp-hardware)

target_compile_definitions(blackchirp-hardware PRIVATE
    BC_HARDWARE_LIBRARY
)

message(STATUS "Hardware: All implementations included for runtime selection")