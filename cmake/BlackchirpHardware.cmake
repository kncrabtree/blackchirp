# BlackchirpHardware.cmake - Hardware layer with dynamic selection system
#
# This module defines the blackchirp-hardware library target with dynamic
# hardware selection based on configuration variables. It replicates the
# functionality of the qmake hardware.pri system.

# Case-insensitive string comparison macro for hardware types
# Usage: if(HARDWARE_EQUALS("${VARIABLE}" "expectedvalue"))
macro(HARDWARE_EQUALS VAR EXPECTED RESULT_VAR)
    string(TOUPPER "${VAR}" __VAR_UPPER)
    string(TOUPPER "${EXPECTED}" __EXPECTED_UPPER)
    if(__VAR_UPPER STREQUAL __EXPECTED_UPPER)
        set(${RESULT_VAR} TRUE)
    else()
        set(${RESULT_VAR} FALSE)
    endif()
endmacro()

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
    
    # Vendor library wrappers (always needed for runtime loading)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/library/vendorlibrary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/library/spectrumlibrary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/library/labjacklibrary.cpp
    
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
    
    # Vendor library wrappers (always needed for runtime loading)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/library/vendorlibrary.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/library/spectrumlibrary.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/library/spectrumconstants.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/library/labjacklibrary.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/library/labjackconstants.h
    
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
    # Set all boolean variables first
    HARDWARE_EQUALS("${BC_FTMWSCOPE}" "virtual" IS_VIRTUAL)
    HARDWARE_EQUALS("${BC_FTMWSCOPE}" "dsa71604c" IS_DSA71604C)
    HARDWARE_EQUALS("${BC_FTMWSCOPE}" "m4i2220x8" IS_M4I2220X8)
    HARDWARE_EQUALS("${BC_FTMWSCOPE}" "dsox92004a" IS_DSOX92004A)
    HARDWARE_EQUALS("${BC_FTMWSCOPE}" "dsov204a" IS_DSOV204A)
    HARDWARE_EQUALS("${BC_FTMWSCOPE}" "dpo71254b" IS_DPO71254B)
    HARDWARE_EQUALS("${BC_FTMWSCOPE}" "mso72004c" IS_MSO72004C)
    HARDWARE_EQUALS("${BC_FTMWSCOPE}" "mso64b" IS_MSO64B)

    if(IS_VIRTUAL)
        add_single_hardware("ftmwdigitizer" "virtualftmwscope" "VirtualFtmwScope" TRUE)
    elseif(IS_DSA71604C)
        add_single_hardware("ftmwdigitizer" "dsa71604c" "Dsa71604c" TRUE)
    elseif(IS_M4I2220X8)
        add_single_hardware("ftmwdigitizer" "m4i2220x8" "M4i2220x8" TRUE)
    elseif(IS_DSOX92004A)
        add_single_hardware("ftmwdigitizer" "dsox92004a" "DSOx92004A" TRUE)
    elseif(IS_DSOV204A)
        add_single_hardware("ftmwdigitizer" "dsov204a" "DSOv204A" TRUE)
    elseif(IS_DPO71254B)
        add_single_hardware("ftmwdigitizer" "dpo71254b" "Dpo71254b" TRUE)
    elseif(IS_MSO72004C)
        add_single_hardware("ftmwdigitizer" "mso72004c" "MSO72004C" TRUE)
    elseif(IS_MSO64B)
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
        HARDWARE_EQUALS("${CLOCK_IMPL}" "fixed" IS_FIXED)

        if(IS_FIXED)
            add_multiple_hardware("clock" "fixedclock" "FixedClock" ${CLOCK_INDEX} TRUE)
        HARDWARE_EQUALS("${CLOCK_IMPL}" "valon5009" IS_VALON5009)

        elseif(IS_VALON5009)
            add_multiple_hardware("clock" "valon5009" "Valon5009" ${CLOCK_INDEX} TRUE)
        HARDWARE_EQUALS("${CLOCK_IMPL}" "valon5015" IS_VALON5015)

        elseif(IS_VALON5015)
            add_multiple_hardware("clock" "valon5015" "Valon5015" ${CLOCK_INDEX} TRUE)
        HARDWARE_EQUALS("${CLOCK_IMPL}" "hp83712b" IS_HP83712B)

        elseif(IS_HP83712B)
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
        HARDWARE_EQUALS("${BC_AWG}" "virtual" IS_VIRTUAL)

        if(IS_VIRTUAL)
            add_single_hardware("chirpsource" "virtualawg" "VirtualAwg" FALSE)
        HARDWARE_EQUALS("${BC_AWG}" "awg70002a" IS_AWG70002A)

        elseif(IS_AWG70002A)
            add_single_hardware("chirpsource" "awg70002a" "AWG70002a" FALSE)
        HARDWARE_EQUALS("${BC_AWG}" "awg7122b" IS_AWG7122B)

        elseif(IS_AWG7122B)
            add_single_hardware("chirpsource" "awg7122b" "AWG7122B" FALSE)
        HARDWARE_EQUALS("${BC_AWG}" "awg5204" IS_AWG5204)

        elseif(IS_AWG5204)
            add_single_hardware("chirpsource" "awg5204" "AWG5204" FALSE)
        HARDWARE_EQUALS("${BC_AWG}" "ad9914" IS_AD9914)

        elseif(IS_AD9914)
            add_single_hardware("chirpsource" "ad9914" "AD9914" FALSE)
        HARDWARE_EQUALS("${BC_AWG}" "m8190" IS_M8190)

        elseif(IS_M8190)
            add_single_hardware("chirpsource" "m8190" "M8190" FALSE)
        HARDWARE_EQUALS("${BC_AWG}" "m8195a" IS_M8195A)

        elseif(IS_M8195A)
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
            HARDWARE_EQUALS("${FC_IMPL}" "virtual" IS_VIRTUAL)

            if(IS_VIRTUAL)
                add_multiple_hardware("flowcontroller" "virtualflowcontroller" "VirtualFlowController" ${FC_INDEX} FALSE)
            HARDWARE_EQUALS("${FC_IMPL}" "mks647c" IS_MKS647C)

            elseif(IS_MKS647C)
                add_multiple_hardware("flowcontroller" "mks647c" "Mks647c" ${FC_INDEX} FALSE)
            HARDWARE_EQUALS("${FC_IMPL}" "mks946" IS_MKS946)

            elseif(IS_MKS946)
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
        HARDWARE_EQUALS("${BC_GPIBCONTROLLER}" "virtual" IS_VIRTUAL)

        if(IS_VIRTUAL)
            add_single_hardware("gpibcontroller" "virtualgpibcontroller" "VirtualGpibController" FALSE)
        HARDWARE_EQUALS("${BC_GPIBCONTROLLER}" "prologixlan" IS_PROLOGIXLAN)

        elseif(IS_PROLOGIXLAN)
            add_single_hardware("gpibcontroller" "prologixgpiblan" "PrologixGpibLan" FALSE)
        HARDWARE_EQUALS("${BC_GPIBCONTROLLER}" "prologixusb" IS_PROLOGIXUSB)

        elseif(IS_PROLOGIXUSB)
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
        # Add U3 wrapper dependency for LabjackU3
        list(APPEND BLACKCHIRP_HARDWARE_DYNAMIC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/ioboard/u3.cpp)
        list(APPEND BLACKCHIRP_HARDWARE_DYNAMIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/ioboard/u3.h)
        
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
            HARDWARE_EQUALS("${IOB_IMPL}" "virtual" IS_VIRTUAL)
            HARDWARE_EQUALS("${IOB_IMPL}" "labjacku3" IS_LABJACKU3)
            
            if(IS_VIRTUAL)
                add_multiple_hardware("ioboard" "virtualioboard" "VirtualIOBoard" ${IOB_INDEX} FALSE)
            elseif(IS_LABJACKU3)
                add_multiple_hardware("ioboard" "labjacku3" "LabjackU3" ${IOB_INDEX} FALSE)
                # Add U3 wrapper dependency
                list(APPEND BLACKCHIRP_HARDWARE_DYNAMIC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/ioboard/u3.cpp)
                list(APPEND BLACKCHIRP_HARDWARE_DYNAMIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/hardware/optional/ioboard/u3.h)
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
            HARDWARE_EQUALS("${PC_IMPL}" "virtual" IS_VIRTUAL)

            if(IS_VIRTUAL)
                add_multiple_hardware("pressurecontroller" "virtualpressurecontroller" "VirtualPressureController" ${PC_INDEX} FALSE)
            HARDWARE_EQUALS("${PC_IMPL}" "intellisysiqplus" IS_INTELLISYSIQPLUS)

            elseif(IS_INTELLISYSIQPLUS)
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
            HARDWARE_EQUALS("${TC_IMPL}" "virtual" IS_VIRTUAL)

            if(IS_VIRTUAL)
                add_multiple_hardware("tempcontroller" "virtualtempcontroller" "VirtualTemperatureController" ${TC_INDEX} FALSE)
            HARDWARE_EQUALS("${TC_IMPL}" "lakeshore218" IS_LAKESHORE218)

            elseif(IS_LAKESHORE218)
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
            HARDWARE_EQUALS("${PGEN_IMPL}" "virtual" IS_VIRTUAL)

            if(IS_VIRTUAL)
                add_multiple_hardware("pulsegenerator" "virtualpulsegenerator" "VirtualPulseGenerator" ${PGEN_INDEX} FALSE)
            HARDWARE_EQUALS("${PGEN_IMPL}" "qc9528" IS_QC9528)

            elseif(IS_QC9528)
                add_multiple_hardware("pulsegenerator" "qcpulsegenerator" "Qc9528" ${PGEN_INDEX} FALSE)
            HARDWARE_EQUALS("${PGEN_IMPL}" "qc9518" IS_QC9518)

            elseif(IS_QC9518)
                add_multiple_hardware("pulsegenerator" "qcpulsegenerator" "Qc9518" ${PGEN_INDEX} FALSE)
            HARDWARE_EQUALS("${PGEN_IMPL}" "qc9214" IS_QC9214)

            elseif(IS_QC9214)
                add_multiple_hardware("pulsegenerator" "qcpulsegenerator" "Qc9214" ${PGEN_INDEX} FALSE)
            HARDWARE_EQUALS("${PGEN_IMPL}" "bnc577" IS_BNC577)

            elseif(IS_BNC577)
                add_multiple_hardware("pulsegenerator" "bnc577" "Bnc577_4" ${PGEN_INDEX} FALSE)
            elseif(PGEN_IMPL STREQUAL "bnc577_8")
                add_multiple_hardware("pulsegenerator" "bnc577" "Bnc577_8" ${PGEN_INDEX} FALSE)
            HARDWARE_EQUALS("${PGEN_IMPL}" "srsdg645" IS_SRSDG645)

            elseif(IS_SRSDG645)
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
            HARDWARE_EQUALS("${BC_LIFSCOPE}" "virtual" IS_VIRTUAL)

            if(IS_VIRTUAL)
                add_single_hardware("lifdigitizer" "virtuallifscope" "VirtualLifScope" TRUE)
            HARDWARE_EQUALS("${BC_LIFSCOPE}" "m4i2211x8" IS_M4I2211X8)

            elseif(IS_M4I2211X8)
                add_single_hardware("lifdigitizer" "m4i2211x8" "M4i2211x8" TRUE)
            HARDWARE_EQUALS("${BC_LIFSCOPE}" "rigolds2302a" IS_RIGOLDS2302A)

            elseif(IS_RIGOLDS2302A)
                add_single_hardware("lifdigitizer" "rigolds2302a" "RigolDS2302A" TRUE)
            else()
                message(FATAL_ERROR "Unknown LIF digitizer implementation: ${BC_LIFSCOPE}")
            endif()
        endif()
        
        # LIF Laser (Required when LIF is enabled)
        if(BC_LIFLASER)
            string(TOUPPER ${BC_LIFLASER} LIFLASER_UPPER)
            HARDWARE_EQUALS("${BC_LIFLASER}" "virtual" IS_VIRTUAL)

            if(IS_VIRTUAL)
                add_single_hardware("liflaser" "virtualliflaser" "VirtualLifLaser" TRUE)
            HARDWARE_EQUALS("${BC_LIFLASER}" "opolette" IS_OPOLETTE)

            elseif(IS_OPOLETTE)
                add_single_hardware("liflaser" "opolette" "Opolette" TRUE)
            HARDWARE_EQUALS("${BC_LIFLASER}" "sirahcobra" IS_SIRAHCOBRA)

            elseif(IS_SIRAHCOBRA)
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

# Hardware-specific library detection (linking happens in final executable)
# LabJack: Now uses dynamic loading via LabjackLibrary wrapper - no compile-time detection needed

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
