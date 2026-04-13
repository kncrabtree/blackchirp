# HardwareValidation.cmake - Validate hardware configuration options
#
# This module provides functions to validate hardware configurations
# and ensure compatibility between selected components.

# Define valid hardware implementations
set(VALID_FTMW_SCOPES 
    "virtual" "DSA71604C" "MSO72004C" "M4i2220x8" 
    "DSOx92004A" "DSOV204A" "MSO64B" "DPO71254B"
)

set(VALID_CLOCKS
    "fixed" "Valon5009" "Valon5015" "HP83712B"
)

set(VALID_AWGS
    "virtual" "AWG70002A" "AWG7122B" "AD9914" "M8195A" "AWG5204"
)

set(VALID_PGENS
    "virtual" "qc9510series" "qc9520series" "qc9210series"
    "bnc577" "srsdg645" "python"
)

set(VALID_FCS
    "virtual" "MKS647C" "MKS946"
)

set(VALID_IOBOARDS
    "virtual" "LabjackU3"
)

set(VALID_GPIBS
    "virtual" "PrologixLAN" "PrologixUSB"
)

set(VALID_PCS
    "virtual" "Intellisys"
)

set(VALID_TCS
    "virtual" "Lakeshore218"
)

set(VALID_LIFSCOPES
    "virtual" "m4i2211x8" "rigolds2302a"
)

set(VALID_LIFLASERS
    "virtual" "opolette" "sirahcobra"
)

# Case-insensitive string comparison function
function(strequal_case_insensitive VAR1 VAR2 RESULT_VAR)
    string(TOUPPER "${VAR1}" VAR1_UPPER)
    string(TOUPPER "${VAR2}" VAR2_UPPER)
    if(VAR1_UPPER STREQUAL VAR2_UPPER)
        set(${RESULT_VAR} TRUE PARENT_SCOPE)
    else()
        set(${RESULT_VAR} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Function to validate a single hardware selection
function(validate_single_hardware HARDWARE_TYPE HARDWARE_VALUE VALID_OPTIONS)
    if(HARDWARE_VALUE)
        # Convert to uppercase for case-insensitive comparison (like qmake build)
        string(TOUPPER "${HARDWARE_VALUE}" HARDWARE_UPPER)
        
        # Convert valid options to uppercase for comparison
        set(VALID_OPTIONS_UPPER "")
        foreach(OPTION IN LISTS VALID_OPTIONS)
            string(TOUPPER "${OPTION}" OPTION_UPPER)
            list(APPEND VALID_OPTIONS_UPPER "${OPTION_UPPER}")
        endforeach()
        
        list(FIND VALID_OPTIONS_UPPER "${HARDWARE_UPPER}" INDEX)
        if(INDEX EQUAL -1)
            message(FATAL_ERROR 
                "Invalid ${HARDWARE_TYPE}: '${HARDWARE_VALUE}'\n"
                "Valid options are: ${VALID_OPTIONS}")
        endif()
    endif()
endfunction()

# Function to validate multiple hardware selections (semicolon-separated list)
function(validate_multiple_hardware HARDWARE_TYPE HARDWARE_LIST VALID_OPTIONS)
    if(HARDWARE_LIST)
        foreach(HARDWARE IN LISTS HARDWARE_LIST)
            if(HARDWARE)
                # Convert to uppercase for case-insensitive comparison (like qmake build)
                string(TOUPPER "${HARDWARE}" HARDWARE_UPPER)
                
                # Convert valid options to uppercase for comparison
                set(VALID_OPTIONS_UPPER "")
                foreach(OPTION IN LISTS VALID_OPTIONS)
                    string(TOUPPER "${OPTION}" OPTION_UPPER)
                    list(APPEND VALID_OPTIONS_UPPER "${OPTION_UPPER}")
                endforeach()
                
                list(FIND VALID_OPTIONS_UPPER "${HARDWARE_UPPER}" INDEX)
                if(INDEX EQUAL -1)
                    message(FATAL_ERROR 
                        "Invalid ${HARDWARE_TYPE}: '${HARDWARE}'\n"
                        "Valid options are: ${VALID_OPTIONS}")
                endif()
            endif()
        endforeach()
    endif()
endfunction()

# Main validation function
function(validate_hardware_configuration)
    message(STATUS "Validating hardware configuration...")
    
    # Validate FTMW scope (required)
    validate_single_hardware("FTMW scope" "${BC_FTMW_SCOPE}" "${VALID_FTMW_SCOPES}")
    
    # Validate clocks (required, at least one)
    if(NOT BC_CLOCKS)
        message(FATAL_ERROR "At least one clock must be specified")
    endif()
    validate_multiple_hardware("clock" "${BC_CLOCKS}" "${VALID_CLOCKS}")
    
    # Validate optional hardware
    validate_single_hardware("AWG" "${BC_AWG}" "${VALID_AWGS}")
    validate_multiple_hardware("pulse generator" "${BC_PGEN}" "${VALID_PGENS}")
    validate_multiple_hardware("flow controller" "${BC_FC}" "${VALID_FCS}")
    validate_multiple_hardware("IO board" "${BC_IOBOARD}" "${VALID_IOBOARDS}")
    validate_single_hardware("GPIB controller" "${BC_GPIB}" "${VALID_GPIBS}")
    validate_multiple_hardware("pressure controller" "${BC_PC}" "${VALID_PCS}")
    validate_multiple_hardware("temperature controller" "${BC_TC}" "${VALID_TCS}")
    
    # Hardware compatibility checks
    validate_hardware_compatibility()
    
    message(STATUS "Hardware configuration validation passed")
endfunction()

# Function to check hardware compatibility
function(validate_hardware_compatibility)
    # Example compatibility checks
    
    # Check if M4i cards are used consistently
    set(M4I_CARDS "M4i2220x8" "M4i2211x8")
    set(USES_M4I FALSE)
    
    if(BC_FTMW_SCOPE IN_LIST M4I_CARDS)
        set(USES_M4I TRUE)
    endif()
    
    # Add more compatibility checks as needed
    if(USES_M4I)
        message(STATUS "M4i hardware detected - ensuring proper configuration")
        # Add M4i-specific validation here
    endif()
    
    # Check for hardware conflicts
    # Example: certain combinations that don't work together
    # Add specific conflict checks here as needed
endfunction()

# Function to get hardware-specific compile definitions
function(get_hardware_definitions HARDWARE_TYPE HARDWARE_VALUE OUTPUT_VAR)
    set(DEFINITIONS "")
    
    if(HARDWARE_VALUE)
        string(TOUPPER "${HARDWARE_VALUE}" HARDWARE_UPPER)
        list(APPEND DEFINITIONS "${HARDWARE_TYPE}_${HARDWARE_UPPER}")
        list(APPEND DEFINITIONS "${HARDWARE_TYPE}=${HARDWARE_VALUE}")
    endif()
    
    set(${OUTPUT_VAR} ${DEFINITIONS} PARENT_SCOPE)
endfunction()

# Function to check if hardware requires additional libraries
function(check_hardware_dependencies)
    set(REQUIRED_LIBS "")
    
    # Check for hardware-specific library requirements
    if(BC_FTMW_SCOPE MATCHES "M4i.*")
        # M4i cards require spectrum driver
        message(STATUS "M4i hardware detected - spectrum driver required")
    endif()
    
    if(BC_IOBOARD MATCHES "LabjackU3")
        # LabJack requires specific libraries
        message(STATUS "LabJack hardware detected - LabJack libraries required")
    endif()
    
    # Add more hardware-specific dependency checks
endfunction()