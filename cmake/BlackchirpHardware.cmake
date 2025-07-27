# BlackchirpHardware.cmake - Hardware layer with dynamic selection system
#
# This module defines the blackchirp-hardware library target with dynamic
# hardware selection based on configuration variables. It replicates the
# functionality of the qmake hardware.pri system.

# Include guard to prevent multiple inclusions
if(BLACKCHIRP_HARDWARE_CMAKE_INCLUDED)
    return()
endif()
set(BLACKCHIRP_HARDWARE_CMAKE_INCLUDED TRUE)

# Include hardware validation
include(${CMAKE_CURRENT_LIST_DIR}/HardwareValidation.cmake)

# ============================================================================
# Hardware Selection Variables
# ============================================================================

# These variables are typically set in the main CMakeLists.txt or via
# configuration files. They determine which hardware implementations to compile.

# Required hardware (single selection)
set(BC_FTMWSCOPE "virtual" CACHE STRING "FTMW digitizer implementation")
set(BC_CLOCKS "fixed;fixed" CACHE STRING "Clock implementations (semicolon-separated)")

# Optional hardware (single selection)
set(BC_AWG "" CACHE STRING "AWG implementation")
set(BC_FLOWCONTROLLER "" CACHE STRING "Flow controller implementation")
set(BC_GPIBCONTROLLER "" CACHE STRING "GPIB controller implementation")
set(BC_IOBOARD "" CACHE STRING "IO board implementation")
set(BC_PRESSURECONTROLLER "" CACHE STRING "Pressure controller implementation")
set(BC_TEMPCONTROLLER "" CACHE STRING "Temperature controller implementation")

# Optional hardware (multiple selection)
set(BC_PGEN "virtual" CACHE STRING "Pulse generator implementations (semicolon-separated)")

# Special configuration flags
option(BC_LIF "Enable LIF-specific hardware support" OFF)
option(BC_ALLHARDWARE "Compile all available hardware implementations" OFF)

# ============================================================================
# Core Hardware Sources (Always Included)
# ============================================================================

set(BLACKCHIRP_HARDWARE_CORE_SOURCES
    # Hardware manager
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/hardwaremanager.cpp
    
    # Communication protocols
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/communication/communicationprotocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/communication/rs232instrument.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/communication/tcpinstrument.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/communication/virtualinstrument.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/communication/gpibinstrument.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/communication/custominstrument.cpp
    
    # Base hardware classes
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/hardwareobject.cpp
    
    # Clock manager (always needed)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/clock/clockmanager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/clock/clock.cpp
    
    # FTMW Scope base class (always needed)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/ftmwscope.cpp
    
    # Optional hardware base classes (always needed for hardware manager)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/chirpsource/awg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pulsegenerator/pulsegenerator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/flowcontroller/flowcontroller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/gpibcontroller/gpibcontroller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/ioboard/ioboard.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pressurecontroller/pressurecontroller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/tempcontroller/temperaturecontroller.cpp
)

set(BLACKCHIRP_HARDWARE_CORE_HEADERS
    # Hardware manager
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/hardwaremanager.h
    
    # Communication protocols
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/communication/communicationprotocol.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/communication/rs232instrument.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/communication/tcpinstrument.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/communication/virtualinstrument.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/communication/gpibinstrument.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/communication/custominstrument.h
    
    # Base hardware classes
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/hardwareobject.h
    
    # Clock manager (always needed)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/clock/clockmanager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/clock/clock.h
    
    # FTMW Scope base class (always needed)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/ftmwdigitizer/ftmwscope.h
    
    # Optional hardware base classes (always needed for hardware manager)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/chirpsource/awg.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pulsegenerator/pulsegenerator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/flowcontroller/flowcontroller.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/gpibcontroller/gpibcontroller.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/ioboard/ioboard.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/pressurecontroller/pressurecontroller.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/tempcontroller/temperaturecontroller.h
)

# ============================================================================
# Dynamic Hardware Source Generation
# ============================================================================

# Initialize variables for dynamic hardware
set(BLACKCHIRP_HARDWARE_DYNAMIC_SOURCES)
set(BLACKCHIRP_HARDWARE_DYNAMIC_HEADERS)
set(BLACKCHIRP_HARDWARE_DEFINITIONS)
set(BLACKCHIRP_HARDWARE_INCLUDES)

# Function to add single-instance hardware (generates class definition)
function(add_single_hardware TYPE IMPL_NAME CLASS_NAME IS_CORE)
    string(TOLOWER ${TYPE} TYPE_LOWER)
    string(TOLOWER ${IMPL_NAME} IMPL_LOWER)
    string(TOUPPER ${TYPE} TYPE_UPPER)
    
    # Determine path based on core vs optional
    if(IS_CORE)
        set(HW_PATH "core")
    else()
        set(HW_PATH "optional")
    endif()
    
    # Add source files
    set(IMPL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/${HW_PATH}/${TYPE_LOWER}/${IMPL_LOWER}.cpp")
    set(IMPL_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/${HW_PATH}/${TYPE_LOWER}/${IMPL_LOWER}.h")
    
    if(EXISTS ${IMPL_SOURCE})
        list(APPEND BLACKCHIRP_HARDWARE_DYNAMIC_SOURCES ${IMPL_SOURCE})
        set(BLACKCHIRP_HARDWARE_DYNAMIC_SOURCES ${BLACKCHIRP_HARDWARE_DYNAMIC_SOURCES} PARENT_SCOPE)
    endif()
    
    if(EXISTS ${IMPL_HEADER})
        list(APPEND BLACKCHIRP_HARDWARE_DYNAMIC_HEADERS ${IMPL_HEADER})
        set(BLACKCHIRP_HARDWARE_DYNAMIC_HEADERS ${BLACKCHIRP_HARDWARE_DYNAMIC_HEADERS} PARENT_SCOPE)
        
        # Add include for auto-generated header
        list(APPEND BLACKCHIRP_HARDWARE_INCLUDES "#include <hardware/${HW_PATH}/${TYPE_LOWER}/${IMPL_LOWER}.h>")
        set(BLACKCHIRP_HARDWARE_INCLUDES ${BLACKCHIRP_HARDWARE_INCLUDES} PARENT_SCOPE)
    endif()
    
    # Add class definition for single-instance hardware
    list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_${TYPE_UPPER}=${CLASS_NAME}")
    set(BLACKCHIRP_HARDWARE_DEFINITIONS ${BLACKCHIRP_HARDWARE_DEFINITIONS} PARENT_SCOPE)
endfunction()

# Function to add hardware sources only (no preprocessor definitions) - used for allhardware mode
function(add_hardware_sources_only TYPE IMPL_NAME IS_CORE)
    string(TOLOWER ${TYPE} TYPE_LOWER)
    string(TOLOWER ${IMPL_NAME} IMPL_LOWER)
    
    # Determine path based on core vs optional
    if(IS_CORE)
        set(HW_PATH "core")
    else()
        set(HW_PATH "optional")
    endif()
    
    # Add source files only
    set(IMPL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/${HW_PATH}/${TYPE_LOWER}/${IMPL_LOWER}.cpp")
    set(IMPL_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/${HW_PATH}/${TYPE_LOWER}/${IMPL_LOWER}.h")
    
    if(EXISTS ${IMPL_SOURCE})
        list(APPEND BLACKCHIRP_HARDWARE_DYNAMIC_SOURCES ${IMPL_SOURCE})
        set(BLACKCHIRP_HARDWARE_DYNAMIC_SOURCES ${BLACKCHIRP_HARDWARE_DYNAMIC_SOURCES} PARENT_SCOPE)
    endif()
    
    if(EXISTS ${IMPL_HEADER})
        list(APPEND BLACKCHIRP_HARDWARE_DYNAMIC_HEADERS ${IMPL_HEADER})
        set(BLACKCHIRP_HARDWARE_DYNAMIC_HEADERS ${BLACKCHIRP_HARDWARE_DYNAMIC_HEADERS} PARENT_SCOPE)
        
        # Add include for auto-generated header
        list(APPEND BLACKCHIRP_HARDWARE_INCLUDES "#include <hardware/${HW_PATH}/${TYPE_LOWER}/${IMPL_LOWER}.h>")
        set(BLACKCHIRP_HARDWARE_INCLUDES ${BLACKCHIRP_HARDWARE_INCLUDES} PARENT_SCOPE)
    endif()
    
    # Note: No preprocessor definitions added - this is the key difference from add_single_hardware
endfunction()

# Function to add multiple-instance hardware (generates flag + indexed definitions)
function(add_multiple_hardware TYPE IMPL_NAME CLASS_NAME INDEX IS_CORE)
    string(TOLOWER ${TYPE} TYPE_LOWER)
    string(TOLOWER ${IMPL_NAME} IMPL_LOWER)
    string(TOUPPER ${TYPE} TYPE_UPPER)
    
    # Determine path based on core vs optional
    if(IS_CORE)
        set(HW_PATH "core")
    else()
        set(HW_PATH "optional")
    endif()
    
    # Add source files
    set(IMPL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/${HW_PATH}/${TYPE_LOWER}/${IMPL_LOWER}.cpp")
    set(IMPL_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/${HW_PATH}/${TYPE_LOWER}/${IMPL_LOWER}.h")
    
    if(EXISTS ${IMPL_SOURCE})
        list(APPEND BLACKCHIRP_HARDWARE_DYNAMIC_SOURCES ${IMPL_SOURCE})
        set(BLACKCHIRP_HARDWARE_DYNAMIC_SOURCES ${BLACKCHIRP_HARDWARE_DYNAMIC_SOURCES} PARENT_SCOPE)
    endif()
    
    if(EXISTS ${IMPL_HEADER})
        list(APPEND BLACKCHIRP_HARDWARE_DYNAMIC_HEADERS ${IMPL_HEADER})
        set(BLACKCHIRP_HARDWARE_DYNAMIC_HEADERS ${BLACKCHIRP_HARDWARE_DYNAMIC_HEADERS} PARENT_SCOPE)
        
        # Add include for auto-generated header
        list(APPEND BLACKCHIRP_HARDWARE_INCLUDES "#include <hardware/${HW_PATH}/${TYPE_LOWER}/${IMPL_LOWER}.h>")
        set(BLACKCHIRP_HARDWARE_INCLUDES ${BLACKCHIRP_HARDWARE_INCLUDES} PARENT_SCOPE)
    endif()
    
    # Add indexed class definition for multi-instance hardware
    list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_${TYPE_UPPER}_${INDEX}=${CLASS_NAME}")
    set(BLACKCHIRP_HARDWARE_DEFINITIONS ${BLACKCHIRP_HARDWARE_DEFINITIONS} PARENT_SCOPE)
endfunction()

# ============================================================================
# Process FTMW Digitizer (Required)
# ============================================================================

if(BC_ALLHARDWARE)
    # In allhardware mode: force selection to virtual and add all sources
    set(BC_FTMWSCOPE "virtual")
    add_single_hardware("ftmwdigitizer" "virtualftmwscope" "VirtualFtmwScope" TRUE)
    
    # Add all other FTMW digitizer sources without preprocessor definitions
    add_hardware_sources_only("ftmwdigitizer" "dsa71604c" TRUE)
    add_hardware_sources_only("ftmwdigitizer" "m4i2220x8" TRUE)
    add_hardware_sources_only("ftmwdigitizer" "dsox92004a" TRUE)
    add_hardware_sources_only("ftmwdigitizer" "dsov204a" TRUE)
    add_hardware_sources_only("ftmwdigitizer" "dpo71254b" TRUE)
    add_hardware_sources_only("ftmwdigitizer" "mso72004c" TRUE)
    add_hardware_sources_only("ftmwdigitizer" "mso64b" TRUE)
else()
    # Normal operation - only add the selected implementation
    string(TOUPPER ${BC_FTMWSCOPE} FTMW_UPPER)
    if(BC_FTMWSCOPE STREQUAL "virtual")
        add_single_hardware("ftmwdigitizer" "virtualftmwscope" "VirtualFtmwScope" TRUE)
    elseif(BC_FTMWSCOPE STREQUAL "dsa71604c")
        add_single_hardware("ftmwdigitizer" "dsa71604c" "Dsa71604c" TRUE)
    elseif(BC_FTMWSCOPE STREQUAL "m4i2220x8")
        add_single_hardware("ftmwdigitizer" "m4i2220x8" "M4i2220x8" TRUE)
    elseif(BC_FTMWSCOPE STREQUAL "dsox92004a")
        add_single_hardware("ftmwdigitizer" "dsox92004a" "DSOx92004A" TRUE)
    elseif(BC_FTMWSCOPE STREQUAL "dsov204a")
        add_single_hardware("ftmwdigitizer" "dsov204a" "DSOv204A" TRUE)
    elseif(BC_FTMWSCOPE STREQUAL "dpo71254b")
        add_single_hardware("ftmwdigitizer" "dpo71254b" "Dpo71254b" TRUE)
    elseif(BC_FTMWSCOPE STREQUAL "mso72004c")
        add_single_hardware("ftmwdigitizer" "mso72004c" "MSO72004C" TRUE)
    elseif(BC_FTMWSCOPE STREQUAL "mso64b")
        add_single_hardware("ftmwdigitizer" "mso64b" "MSO64B" TRUE)
    else()
        message(FATAL_ERROR "Unknown FTMW digitizer: ${BC_FTMWSCOPE}")
    endif()
endif()

# ============================================================================
# Process Clocks (Required, Multiple)
# ============================================================================

if(BC_ALLHARDWARE)
    # In allhardware mode: force selection to fixed and add all sources
    set(BC_CLOCKS "fixed;fixed")
    list(LENGTH BC_CLOCKS NUM_CLOCKS)
    list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_NUM_CLOCKS=${NUM_CLOCKS}")
    
    # Add fixed clock with definitions (since it's selected)
    add_multiple_hardware("clock" "fixedclock" "FixedClock" 0 TRUE)
    add_multiple_hardware("clock" "fixedclock" "FixedClock" 1 TRUE)
    
    # Add all other clock sources without preprocessor definitions
    add_hardware_sources_only("clock" "valon5009" TRUE)
    add_hardware_sources_only("clock" "valon5015" TRUE)
    add_hardware_sources_only("clock" "hp83712b" TRUE)
else()
    # Normal operation - process the selected implementations
    list(LENGTH BC_CLOCKS NUM_CLOCKS)
    if(NUM_CLOCKS EQUAL 0)
        message(FATAL_ERROR "At least one clock implementation must be specified")
    endif()

    list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_NUM_CLOCKS=${NUM_CLOCKS}")

    set(CLOCK_INDEX 0)
    foreach(CLOCK_IMPL IN LISTS BC_CLOCKS)
        string(TOUPPER ${CLOCK_IMPL} CLOCK_UPPER)
        if(CLOCK_IMPL STREQUAL "fixed")
            add_multiple_hardware("clock" "fixedclock" "FixedClock" ${CLOCK_INDEX} TRUE)
        elseif(CLOCK_IMPL STREQUAL "valon5009")
            add_multiple_hardware("clock" "valon5009" "Valon5009" ${CLOCK_INDEX} TRUE)
        elseif(CLOCK_IMPL STREQUAL "valon5015")
            add_multiple_hardware("clock" "valon5015" "Valon5015" ${CLOCK_INDEX} TRUE)
        elseif(CLOCK_IMPL STREQUAL "hp83712b")
            add_multiple_hardware("clock" "hp83712b" "HP83712B" ${CLOCK_INDEX} TRUE)
        else()
            message(FATAL_ERROR "Unknown clock implementation: ${CLOCK_IMPL}")
        endif()
        math(EXPR CLOCK_INDEX "${CLOCK_INDEX} + 1")
    endforeach()
endif()

# ============================================================================
# Process Optional Single Hardware
# ============================================================================

# AWG (Arbitrary Waveform Generator - was chirpsource)
if(BC_AWG OR BC_ALLHARDWARE)
    if(BC_ALLHARDWARE)
        # In allhardware mode: force selection to virtual and add all sources
        set(BC_AWG "virtual")
        add_single_hardware("chirpsource" "virtualawg" "VirtualAwg" FALSE)
        
        # Add all other AWG sources without preprocessor definitions
        add_hardware_sources_only("chirpsource" "awg70002a" FALSE)
        add_hardware_sources_only("chirpsource" "awg7122b" FALSE)
        add_hardware_sources_only("chirpsource" "awg5204" FALSE)
        add_hardware_sources_only("chirpsource" "ad9914" FALSE)
        add_hardware_sources_only("chirpsource" "m8190" FALSE)
        add_hardware_sources_only("chirpsource" "m8195a" FALSE)
    else()
        # Normal operation - only add the selected implementation
        string(TOUPPER ${BC_AWG} AWG_UPPER)
        if(BC_AWG STREQUAL "virtual")
            add_single_hardware("chirpsource" "virtualawg" "VirtualAwg" FALSE)
        elseif(BC_AWG STREQUAL "awg70002a")
            add_single_hardware("chirpsource" "awg70002a" "AWG70002a" FALSE)
        elseif(BC_AWG STREQUAL "awg7122b")
            add_single_hardware("chirpsource" "awg7122b" "AWG7122B" FALSE)
        elseif(BC_AWG STREQUAL "awg5204")
            add_single_hardware("chirpsource" "awg5204" "AWG5204" FALSE)
        elseif(BC_AWG STREQUAL "ad9914")
            add_single_hardware("chirpsource" "ad9914" "AD9914" FALSE)
        elseif(BC_AWG STREQUAL "m8190")
            add_single_hardware("chirpsource" "m8190" "M8190" FALSE)
        elseif(BC_AWG STREQUAL "m8195a")
            add_single_hardware("chirpsource" "m8195a" "M8195A" FALSE)
        else()
            message(FATAL_ERROR "Unknown AWG implementation: ${BC_AWG}")
        endif()
    endif()
endif()

# Flow Controller (handles multiple like pulse generators)
if(BC_FLOWCONTROLLER OR BC_ALLHARDWARE)
    if(BC_ALLHARDWARE)
        # In allhardware mode: force selection to virtual and add all sources
        set(BC_FLOWCONTROLLER "virtual")
        set(BC_FLOWCONTROLLER_LIST ${BC_FLOWCONTROLLER})
        list(LENGTH BC_FLOWCONTROLLER_LIST NUM_FC)
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_NUM_FLOWCONTROLLER=${NUM_FC}")
        
        # Add virtual flow controller with definitions (since it's selected)
        add_multiple_hardware("flowcontroller" "virtualflowcontroller" "VirtualFlowController" 0 FALSE)
        
        # Add all other flow controller sources without preprocessor definitions
        add_hardware_sources_only("flowcontroller" "mks647c" FALSE)
        add_hardware_sources_only("flowcontroller" "mks946" FALSE)
        
        # Add ifdef guard definition for conditional compilation throughout codebase
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_FLOWCONTROLLER=1")
    else()
        # Normal operation - process the selected implementations
        # Convert semicolon-separated string to CMake list
        set(BC_FLOWCONTROLLER_LIST ${BC_FLOWCONTROLLER})
        
        list(LENGTH BC_FLOWCONTROLLER_LIST NUM_FC)
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_NUM_FLOWCONTROLLER=${NUM_FC}")
        
        set(FC_INDEX 0)
        foreach(FC_IMPL IN LISTS BC_FLOWCONTROLLER_LIST)
            string(TOUPPER ${FC_IMPL} FC_UPPER)
            if(FC_IMPL STREQUAL "virtual")
                add_multiple_hardware("flowcontroller" "virtualflowcontroller" "VirtualFlowController" ${FC_INDEX} FALSE)
            elseif(FC_IMPL STREQUAL "mks647c")
                add_multiple_hardware("flowcontroller" "mks647c" "Mks647c" ${FC_INDEX} FALSE)
            elseif(FC_IMPL STREQUAL "mks946")
                add_multiple_hardware("flowcontroller" "mks946" "Mks946" ${FC_INDEX} FALSE)
            else()
                message(FATAL_ERROR "Unknown flow controller implementation: ${FC_IMPL}")
            endif()
            math(EXPR FC_INDEX "${FC_INDEX} + 1")
        endforeach()
        # Add ifdef guard definition for conditional compilation throughout codebase
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_FLOWCONTROLLER=1")
    endif()
endif()

# GPIB Controller
if(BC_GPIBCONTROLLER OR BC_ALLHARDWARE)
    if(BC_ALLHARDWARE)
        # In allhardware mode: force selection to virtual and add all sources
        set(BC_GPIBCONTROLLER "virtual")
        add_single_hardware("gpibcontroller" "virtualgpibcontroller" "VirtualGpibController" FALSE)
        
        # Add all other GPIB controller sources without preprocessor definitions
        add_hardware_sources_only("gpibcontroller" "prologixgpiblan" FALSE)
        add_hardware_sources_only("gpibcontroller" "prologixgpibusb" FALSE)
    else()
        # Normal operation - only add the selected implementation
        string(TOUPPER ${BC_GPIBCONTROLLER} GPIB_UPPER)
        if(BC_GPIBCONTROLLER STREQUAL "virtual")
            add_single_hardware("gpibcontroller" "virtualgpibcontroller" "VirtualGpibController" FALSE)
        elseif(BC_GPIBCONTROLLER STREQUAL "prologixlan")
            add_single_hardware("gpibcontroller" "prologixgpiblan" "PrologixGpibLan" FALSE)
        elseif(BC_GPIBCONTROLLER STREQUAL "prologixusb")
            add_single_hardware("gpibcontroller" "prologixgpibusb" "PrologixGpibUsb" FALSE)
        else()
            message(FATAL_ERROR "Unknown GPIB controller implementation: ${BC_GPIBCONTROLLER}")
        endif()
    endif()
endif()

# IO Board (handles multiple)
if(BC_IOBOARD OR BC_ALLHARDWARE)
    if(BC_ALLHARDWARE)
        # In allhardware mode: force selection to virtual and add all sources
        set(BC_IOBOARD "virtual")
        set(BC_IOBOARD_LIST ${BC_IOBOARD})
        list(LENGTH BC_IOBOARD_LIST NUM_IOB)
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_NUM_IOBOARD=${NUM_IOB}")
        
        # Add virtual IO board with definitions (since it's selected)
        add_multiple_hardware("ioboard" "virtualioboard" "VirtualIOBoard" 0 FALSE)
        
        # Add all other IO board sources without preprocessor definitions
        add_hardware_sources_only("ioboard" "labjacku3" FALSE)
        
        # Add ifdef guard definition for conditional compilation throughout codebase
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_IOBOARD=1")
    else()
        # Normal operation - process the selected implementations
        # Convert semicolon-separated string to CMake list
        set(BC_IOBOARD_LIST ${BC_IOBOARD})
        
        list(LENGTH BC_IOBOARD_LIST NUM_IOB)
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_NUM_IOBOARD=${NUM_IOB}")
        
        set(IOB_INDEX 0)
        foreach(IOB_IMPL IN LISTS BC_IOBOARD_LIST)
            string(TOUPPER ${IOB_IMPL} IOB_UPPER)
            if(IOB_IMPL STREQUAL "virtual")
                add_multiple_hardware("ioboard" "virtualioboard" "VirtualIOBoard" ${IOB_INDEX} FALSE)
            elseif(IOB_IMPL STREQUAL "labjacku3")
                add_multiple_hardware("ioboard" "labjacku3" "LabjackU3" ${IOB_INDEX} FALSE)
            else()
                message(FATAL_ERROR "Unknown IO board implementation: ${IOB_IMPL}")
            endif()
            math(EXPR IOB_INDEX "${IOB_INDEX} + 1")
        endforeach()
        # Add ifdef guard definition for conditional compilation throughout codebase
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_IOBOARD=1")
    endif()
endif()

# Pressure Controller (handles multiple)
if(BC_PRESSURECONTROLLER OR BC_ALLHARDWARE)
    if(BC_ALLHARDWARE)
        # In allhardware mode: force selection to virtual and add all sources
        set(BC_PRESSURECONTROLLER "virtual")
        set(BC_PRESSURECONTROLLER_LIST ${BC_PRESSURECONTROLLER})
        list(LENGTH BC_PRESSURECONTROLLER_LIST NUM_PC)
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_NUM_PCONTROLLER=${NUM_PC}")
        
        # Add virtual pressure controller with definitions (since it's selected)
        add_multiple_hardware("pressurecontroller" "virtualpressurecontroller" "VirtualPressureController" 0 FALSE)
        
        # Add all other pressure controller sources without preprocessor definitions
        add_hardware_sources_only("pressurecontroller" "intellisysiqplus" FALSE)
        
        # Add ifdef guard definition for conditional compilation throughout codebase
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_PCONTROLLER=1")
    else()
        # Normal operation - process the selected implementations
        # Convert semicolon-separated string to CMake list
        set(BC_PRESSURECONTROLLER_LIST ${BC_PRESSURECONTROLLER})
        
        list(LENGTH BC_PRESSURECONTROLLER_LIST NUM_PC)
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_NUM_PCONTROLLER=${NUM_PC}")
        
        set(PC_INDEX 0)
        foreach(PC_IMPL IN LISTS BC_PRESSURECONTROLLER_LIST)
            string(TOUPPER ${PC_IMPL} PC_UPPER)
            if(PC_IMPL STREQUAL "virtual")
                add_multiple_hardware("pressurecontroller" "virtualpressurecontroller" "VirtualPressureController" ${PC_INDEX} FALSE)
            elseif(PC_IMPL STREQUAL "intellisysiqplus")
                add_multiple_hardware("pressurecontroller" "intellisysiqplus" "IntellisysIQPlus" ${PC_INDEX} FALSE)
            else()
                message(FATAL_ERROR "Unknown pressure controller implementation: ${PC_IMPL}")
            endif()
            math(EXPR PC_INDEX "${PC_INDEX} + 1")
        endforeach()
        # Add ifdef guard definition for conditional compilation throughout codebase
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_PCONTROLLER=1")
    endif()
endif()

# Temperature Controller (handles multiple)
if(BC_TEMPCONTROLLER OR BC_ALLHARDWARE)
    if(BC_ALLHARDWARE)
        # In allhardware mode: force selection to virtual and add all sources
        set(BC_TEMPCONTROLLER "virtual")
        set(BC_TEMPCONTROLLER_LIST ${BC_TEMPCONTROLLER})
        list(LENGTH BC_TEMPCONTROLLER_LIST NUM_TC)
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_NUM_TEMPCONTROLLER=${NUM_TC}")
        
        # Add virtual temperature controller with definitions (since it's selected)
        add_multiple_hardware("tempcontroller" "virtualtempcontroller" "VirtualTemperatureController" 0 FALSE)
        
        # Add all other temperature controller sources without preprocessor definitions
        add_hardware_sources_only("tempcontroller" "lakeshore218" FALSE)
        
        # Add ifdef guard definition for conditional compilation throughout codebase
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_TEMPCONTROLLER=1")
    else()
        # Normal operation - process the selected implementations
        # Convert semicolon-separated string to CMake list
        set(BC_TEMPCONTROLLER_LIST ${BC_TEMPCONTROLLER})
        
        list(LENGTH BC_TEMPCONTROLLER_LIST NUM_TC)
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_NUM_TEMPCONTROLLER=${NUM_TC}")
        
        set(TC_INDEX 0)
        foreach(TC_IMPL IN LISTS BC_TEMPCONTROLLER_LIST)
            string(TOUPPER ${TC_IMPL} TC_UPPER)
            if(TC_IMPL STREQUAL "virtual")
                add_multiple_hardware("tempcontroller" "virtualtempcontroller" "VirtualTemperatureController" ${TC_INDEX} FALSE)
            elseif(TC_IMPL STREQUAL "lakeshore218")
                add_multiple_hardware("tempcontroller" "lakeshore218" "Lakeshore218" ${TC_INDEX} FALSE)
            else()
                message(FATAL_ERROR "Unknown temperature controller implementation: ${TC_IMPL}")
            endif()
            math(EXPR TC_INDEX "${TC_INDEX} + 1")
        endforeach()
        # Add ifdef guard definition for conditional compilation throughout codebase
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_TEMPCONTROLLER=1")
    endif()
endif()

# ============================================================================
# Process Optional Multiple Hardware
# ============================================================================

# Pulse Generators
if(BC_PGEN OR BC_ALLHARDWARE)
    if(BC_ALLHARDWARE)
        # In allhardware mode: force selection to virtual and add all sources
        set(BC_PGEN "virtual")
        list(LENGTH BC_PGEN NUM_PGEN)
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_NUM_PGEN=${NUM_PGEN}")
        
        # Add virtual pulse generator with definitions (since it's selected)
        add_multiple_hardware("pulsegenerator" "virtualpulsegenerator" "VirtualPulseGenerator" 0 FALSE)
        
        # Add all other pulse generator sources without preprocessor definitions
        add_hardware_sources_only("pulsegenerator" "qcpulsegenerator" FALSE)
        add_hardware_sources_only("pulsegenerator" "bnc577" FALSE)
        add_hardware_sources_only("pulsegenerator" "srsdg645" FALSE)
        
        # Add ifdef guard definition for conditional compilation throughout codebase
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_PGEN=1")
    else()
        # Normal operation - process the selected implementations
        list(LENGTH BC_PGEN NUM_PGEN)
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_NUM_PGEN=${NUM_PGEN}")
        
        set(PGEN_INDEX 0)
        foreach(PGEN_IMPL IN LISTS BC_PGEN)
            string(TOUPPER ${PGEN_IMPL} PGEN_UPPER)
            if(PGEN_IMPL STREQUAL "virtual")
                add_multiple_hardware("pulsegenerator" "virtualpulsegenerator" "VirtualPulseGenerator" ${PGEN_INDEX} FALSE)
            elseif(PGEN_IMPL STREQUAL "qc9528")
                add_multiple_hardware("pulsegenerator" "qcpulsegenerator" "Qc9528" ${PGEN_INDEX} FALSE)
            elseif(PGEN_IMPL STREQUAL "qc9518")
                add_multiple_hardware("pulsegenerator" "qcpulsegenerator" "Qc9518" ${PGEN_INDEX} FALSE)
            elseif(PGEN_IMPL STREQUAL "qc9214")
                add_multiple_hardware("pulsegenerator" "qcpulsegenerator" "Qc9214" ${PGEN_INDEX} FALSE)
            elseif(PGEN_IMPL STREQUAL "bnc577")
                add_multiple_hardware("pulsegenerator" "bnc577" "Bnc577_4" ${PGEN_INDEX} FALSE)
            elseif(PGEN_IMPL STREQUAL "bnc577_8")
                add_multiple_hardware("pulsegenerator" "bnc577" "Bnc577_8" ${PGEN_INDEX} FALSE)
            elseif(PGEN_IMPL STREQUAL "srsdg645")
                add_multiple_hardware("pulsegenerator" "srsdg645" "SRSDG645" ${PGEN_INDEX} FALSE)
            else()
                message(FATAL_ERROR "Unknown pulse generator implementation: ${PGEN_IMPL}")
            endif()
            math(EXPR PGEN_INDEX "${PGEN_INDEX} + 1")
        endforeach()
        # Add ifdef guard definition for conditional compilation throughout codebase
        list(APPEND BLACKCHIRP_HARDWARE_DEFINITIONS "BC_PGEN=1")
    endif()
endif()

# ============================================================================
# Process LIF Hardware (Core when enabled)
# ============================================================================

if(BC_ENABLE_LIF)
    if(BC_ALLHARDWARE)
        # In allhardware mode with LIF enabled: force selections to virtual, add all sources
        set(BC_LIFSCOPE "virtual")
        set(BC_LIFLASER "virtual")
        
        # Add virtual implementations with definitions (since they're selected)
        add_single_hardware("lifdigitizer" "virtuallifscope" "VirtualLifScope" TRUE)
        add_single_hardware("liflaser" "virtualliflaser" "VirtualLifLaser" TRUE)
        
        # Add all other LIF hardware sources without preprocessor definitions
        # NOTE: m4i2211x8 requires Spectrum driver SDK headers to compile
        add_hardware_sources_only("lifdigitizer" "m4i2211x8" TRUE)
        add_hardware_sources_only("lifdigitizer" "rigolds2302a" TRUE)
        add_hardware_sources_only("liflaser" "opolette" TRUE)
        add_hardware_sources_only("liflaser" "sirahcobra" TRUE)
    else()
        # Normal operation - process the selected implementations
        # LIF Digitizer/Oscilloscope (Required when LIF is enabled)
        if(BC_LIFSCOPE)
            string(TOUPPER ${BC_LIFSCOPE} LIFSCOPE_UPPER)
            if(BC_LIFSCOPE STREQUAL "virtual")
                add_single_hardware("lifdigitizer" "virtuallifscope" "VirtualLifScope" TRUE)
            elseif(BC_LIFSCOPE STREQUAL "m4i2211x8")
                add_single_hardware("lifdigitizer" "m4i2211x8" "M4i2211x8" TRUE)
            elseif(BC_LIFSCOPE STREQUAL "rigolds2302a")
                add_single_hardware("lifdigitizer" "rigolds2302a" "RigolDS2302A" TRUE)
            else()
                message(FATAL_ERROR "Unknown LIF digitizer implementation: ${BC_LIFSCOPE}")
            endif()
        endif()
        
        # LIF Laser (Required when LIF is enabled)
        if(BC_LIFLASER)
            string(TOUPPER ${BC_LIFLASER} LIFLASER_UPPER)
            if(BC_LIFLASER STREQUAL "virtual")
                add_single_hardware("liflaser" "virtualliflaser" "VirtualLifLaser" TRUE)
            elseif(BC_LIFLASER STREQUAL "opolette")
                add_single_hardware("liflaser" "opolette" "Opolette" TRUE)
            elseif(BC_LIFLASER STREQUAL "sirahcobra")
                add_single_hardware("liflaser" "sirahcobra" "SirahCobra" TRUE)
            else()
                message(FATAL_ERROR "Unknown LIF laser implementation: ${BC_LIFLASER}")
            endif()
        endif()
    endif()
    
    # Add LIF base classes to core sources (always needed when LIF is enabled)
    list(APPEND BLACKCHIRP_HARDWARE_CORE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/lifdigitizer/lifscope.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/liflaser/liflaser.cpp
    )
    
    list(APPEND BLACKCHIRP_HARDWARE_CORE_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/lifdigitizer/lifscope.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/core/liflaser/liflaser.h
    )
endif()

# ============================================================================
# Clean Old QMake Generated Files
# ============================================================================

# Remove old qmake-generated files that would conflict with CMake versions
set(OLD_QMAKE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/hw_h.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/opthw_h.h"
)

foreach(OLD_FILE ${OLD_QMAKE_FILES})
    if(EXISTS ${OLD_FILE})
        message(STATUS "Removing old qmake-generated file: ${OLD_FILE}")
        file(REMOVE ${OLD_FILE})
    endif()
endforeach()

# ============================================================================
# Generate Auto-Include Headers
# ============================================================================

# Generate hw_h.h (core hardware includes)
string(JOIN "\n" HW_INCLUDES_CONTENT ${BLACKCHIRP_HARDWARE_INCLUDES})
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/src/hardware/hw_h.h" 
    "// Auto-generated hardware includes - DO NOT EDIT\n"
    "#ifndef HW_H_H\n"
    "#define HW_H_H\n\n"
    "${HW_INCLUDES_CONTENT}\n\n"
    "#endif // HW_H_H\n"
)

# Generate opthw_h.h (optional hardware includes)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/src/hardware/opthw_h.h"
    "// Auto-generated optional hardware includes - DO NOT EDIT\n"
    "#ifndef OPTHW_H_H\n"
    "#define OPTHW_H_H\n\n"
    "// Optional hardware includes are already included in hw_h.h\n\n"
    "#endif // OPTHW_H_H\n"
)

# ============================================================================
# Combine All Hardware Sources
# ============================================================================

set(BLACKCHIRP_HARDWARE_ALL_SOURCES
    ${BLACKCHIRP_HARDWARE_CORE_SOURCES}
    ${BLACKCHIRP_HARDWARE_DYNAMIC_SOURCES}
)

set(BLACKCHIRP_HARDWARE_ALL_HEADERS
    ${BLACKCHIRP_HARDWARE_CORE_HEADERS}
    ${BLACKCHIRP_HARDWARE_DYNAMIC_HEADERS}
    "${CMAKE_CURRENT_BINARY_DIR}/src/hardware/hw_h.h"
    "${CMAKE_CURRENT_BINARY_DIR}/src/hardware/opthw_h.h"
)

# ============================================================================
# Create Hardware Library Target
# ============================================================================

# Create the blackchirp-hardware library
add_library(blackchirp-hardware STATIC
    ${BLACKCHIRP_HARDWARE_ALL_SOURCES}
    ${BLACKCHIRP_HARDWARE_ALL_HEADERS}
)

# Add alias for consistent naming
add_library(Blackchirp::Hardware ALIAS blackchirp-hardware)

# ============================================================================
# Target Properties and Configuration
# ============================================================================

# Set target properties
set_target_properties(blackchirp-hardware PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "blackchirp-hardware"
    EXPORT_NAME "Hardware"
)

# Include directories
target_include_directories(blackchirp-hardware
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
)

# ============================================================================
# Dependencies and Linking
# ============================================================================

target_link_libraries(blackchirp-hardware
    PUBLIC
        Qt6::Core
        Qt6::SerialPort
        Qt6::Network
        Blackchirp::Data
    PRIVATE
        GSL::gsl
        GSL::gslcblas
)

# Add system libraries if needed
if(UNIX)
    target_link_libraries(blackchirp-hardware PRIVATE m)
endif()

# ============================================================================
# Compile Definitions
# ============================================================================

# Add version and configuration definitions
add_blackchirp_definitions(blackchirp-hardware)

# Add hardware-specific definitions
target_compile_definitions(blackchirp-hardware PRIVATE
    BC_HARDWARE_LIBRARY
    ${BLACKCHIRP_HARDWARE_DEFINITIONS}
)

# Special configuration modes
if(BC_LIF)
    target_compile_definitions(blackchirp-hardware PRIVATE BC_LIF)
endif()

if(BC_ALLHARDWARE)
    target_compile_definitions(blackchirp-hardware PRIVATE BC_ALLHARDWARE)
endif()

# ============================================================================
# Status Information
# ============================================================================

message(STATUS "Blackchirp Hardware Layer Configuration:")
message(STATUS "  FTMW Digitizer: ${BC_FTMWSCOPE}")
message(STATUS "  Clocks: ${BC_CLOCKS} (${NUM_CLOCKS} total)")
if(BC_AWG)
    message(STATUS "  AWG: ${BC_AWG}")
endif()
if(BC_PGEN)
    list(LENGTH BC_PGEN PGEN_COUNT)
    message(STATUS "  Pulse Generators: ${BC_PGEN} (${PGEN_COUNT} total)")
endif()
if(BC_FLOWCONTROLLER)
    message(STATUS "  Flow Controller: ${BC_FLOWCONTROLLER}")
endif()
if(BC_GPIBCONTROLLER)
    message(STATUS "  GPIB Controller: ${BC_GPIBCONTROLLER}")
endif()
if(BC_IOBOARD)
    message(STATUS "  IO Board: ${BC_IOBOARD}")
endif()
if(BC_PRESSURECONTROLLER)
    message(STATUS "  Pressure Controller: ${BC_PRESSURECONTROLLER}")
endif()
if(BC_TEMPCONTROLLER)
    message(STATUS "  Temperature Controller: ${BC_TEMPCONTROLLER}")
endif()
if(BC_LIF)
    message(STATUS "  LIF Support: ENABLED")
endif()
if(BC_ALLHARDWARE)
    message(STATUS "  All Hardware Mode: ENABLED")
endif()
