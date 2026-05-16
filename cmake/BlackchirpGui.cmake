# BlackchirpGui.cmake - GUI layer components for Blackchirp
#
# This module defines the blackchirp-gui library target containing:
# - Main window and dialogs
# - Experiment setup pages and wizards
# - Plotting components and QWT integration
# - Custom widgets and controls
# - Overlay management system
# - Theme and styling components
# - Data models for GUI components

# Include guard to prevent multiple inclusions
if(BLACKCHIRP_GUI_CMAKE_INCLUDED)
    return()
endif()
set(BLACKCHIRP_GUI_CMAKE_INCLUDED TRUE)

# ============================================================================
# GUI Layer Source Files
# ============================================================================

set(BLACKCHIRP_GUI_SOURCES
    # Main window
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/mainwindow.cpp
    
    # Dialogs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/aboutdialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/addprofiledialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/applicationconfigdialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/batchsequencedialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/bcsavepathdialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/communicationdialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/crashreportdialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/hwarrayeditdialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/hwdialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/peaklistexportdialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/quickexptdialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/ftmwconfigdialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwconfigwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/runtimehardwareconfigdialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/updateavailabledialog.cpp

    # Experiment setup pages
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/drscanconfigwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentconfigpage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentflowconfigpage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentftmwconfigpage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentioboardconfigpage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/loscanconfigwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentpressurecontrollerconfigpage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentpulsegenconfigpage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentsetupdialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentsummarypage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimenttemperaturecontrollerconfigpage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimenttypepage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentvalidatorconfigpage.cpp
    
    # Overlay system
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/unifiedoverlaydialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/unifiedoverlaywidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/overlaytypespecificwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/bcexpoverlaywidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/catalogoverlaywidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/genericxyoverlaywidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/overlaybaseoptionswidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/overlaymanagerwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/overlayconfiguredelegate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/overlaycheckboxdelegate.cpp
    
    # Plotting components
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/blackchirpplotcurve.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/chirpconfigplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/curveappearancewidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/curveappearancepresetmanager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/presetsavedialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/curvefactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/customtracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/customzoomer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/fidplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/ftplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/mainftplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/pulseplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/trackingplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/zoompanplot.cpp
    
    # Custom widgets
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/auxdataviewwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/cellwidgethelpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/chirpconfigwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/clockdisplaybox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/digitizerconfigwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/experimentsummarywidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/experimentviewwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwacquisitionpanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwdigitizerconfigwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwplotpanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwprocessingpanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwviewwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/gascontrolwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/gasflowdisplaywidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/hardwarestatusbox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ioboardconfigwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/led.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/peakfindwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/pressurecontrolwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/pressurestatusbox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/protocolwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/pulseconfigwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/rs232protocolwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/tcpprotocolwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/virtualprotocolwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/customprotocolwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/gpibprotocolwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/pulsestatusbox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/rfconfigwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/hwsettingswidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/settingstable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/scientificspinbox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/scientificinputwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/util/numericformat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/temperaturecontrolwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/temperaturestatusbox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/toolbarwidgetaction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/bcsavepathwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/pythonhardwarecontrolwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/librarystatuswidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/pythonsettingswidget.cpp

    # Styling and theming
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/style/themecolors.cpp
    
    # Data models (moved from data layer - they belong in GUI)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/chirptablemodel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/clocktablemodel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/exptsummarymodel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/markertablemodel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/overlaytablemodel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/peaklistmodel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/peaklistfilterproxymodel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/validationmodel.cpp
)

set(BLACKCHIRP_GUI_HEADERS
    # Main window
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/mainwindow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/mainwindow_ui.h
    
    # Dialogs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/aboutdialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/addprofiledialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/applicationconfigdialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/batchsequencedialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/bcsavepathdialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/communicationdialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/crashreportdialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/hwarrayeditdialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/hwdialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/peaklistexportdialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/ftmwconfigdialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/quickexptdialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/updateavailabledialog.h

    # Experiment setup pages
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/drscanconfigwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentconfigpage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentflowconfigpage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentftmwconfigpage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentioboardconfigpage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/loscanconfigwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentpressurecontrollerconfigpage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentpulsegenconfigpage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentsetupdialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentsummarypage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimenttemperaturecontrollerconfigpage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimenttypepage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/expsetup/experimentvalidatorconfigpage.h
    
    # Overlay system
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/unifiedoverlaydialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/unifiedoverlaywidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/overlaytypespecificwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/bcexpoverlaywidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/catalogoverlaywidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/genericxyoverlaywidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/overlaybaseoptionswidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/overlaymanagerwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/overlayconfiguredelegate.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/overlay/overlaycheckboxdelegate.h
    
    # Plotting components
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/blackchirpplotcurve.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/chirpconfigplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/curveappearancewidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/curveappearancepresetmanager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/presetsavedialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/curvefactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/customtracker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/customzoomer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/fidplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/ftplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/mainftplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/pulseplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/trackingplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/zoompanplot.h
    
    # Custom widgets
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/auxdataviewwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/cellwidgethelpers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/chirpconfigwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/clockdisplaybox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/digitizerconfigwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/enumcombobox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/experimentsummarywidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/experimentviewwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwconfigwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwacquisitionpanel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwdigitizerconfigwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwplotpanel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwprocessingpanel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwviewwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/gascontrolwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/gasflowdisplaywidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/hardwarestatusbox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ioboardconfigwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/led.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/peakfindwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/pressurecontrolwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/pressurestatusbox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/protocolwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/pulseconfigwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/rs232protocolwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/tcpprotocolwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/virtualprotocolwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/customprotocolwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/gpibprotocolwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/pulsestatusbox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/rfconfigwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/hwsettingswidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/settingstable.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/scientificspinbox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/scientificinputwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/util/numericformat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/temperaturecontrolwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/temperaturestatusbox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/toolbarwidgetaction.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/bcsavepathwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/pythonhardwarecontrolwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/librarystatuswidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/pythonsettingswidget.h

    # Styling and theming
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/style/themecolors.h
    
    # Data models (moved from data layer - they belong in GUI)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/chirptablemodel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/clocktablemodel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/exptsummarymodel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/markertablemodel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/overlaytablemodel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/peaklistmodel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/peaklistfilterproxymodel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/validationmodel.h
)

# UI Forms
set(BLACKCHIRP_GUI_FORMS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/communicationdialog.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/chirpconfigwidget.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/digitizerconfigwidget.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/rfconfigwidget.ui
)

# ============================================================================
# LIF GUI Components (Always included for runtime configuration)
# ============================================================================

# LIF GUI sources are always included for runtime configuration
list(APPEND BLACKCHIRP_GUI_SOURCES
    # LIF-specific GUI components
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/experimentlifconfigpage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifcontrolwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifdisplaywidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/liflaserstatusbox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/liflaserwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifprocessingwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifsliceplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifspectrogramplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/liftraceplot.cpp
)

list(APPEND BLACKCHIRP_GUI_HEADERS
    # LIF-specific GUI headers
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/experimentlifconfigpage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifcontrolwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifdisplaywidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/liflaserstatusbox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/liflaserwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifprocessingwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifsliceplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifspectrogramplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/liftraceplot.h
)

# ============================================================================
# Create GUI Library Target
# ============================================================================

# Create the blackchirp-gui library
add_library(blackchirp-gui STATIC
    ${BLACKCHIRP_GUI_SOURCES}
    ${BLACKCHIRP_GUI_HEADERS}
    ${BLACKCHIRP_GUI_FORMS}
)

# Add alias for consistent naming
add_library(Blackchirp::Gui ALIAS blackchirp-gui)

# ============================================================================
# Target Properties and Configuration
# ============================================================================

# Set target properties
set_target_properties(blackchirp-gui PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "blackchirp-gui"
    EXPORT_NAME "Gui"
)

# Include directories
target_include_directories(blackchirp-gui
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
)

# ============================================================================
# Dependencies and Linking
# ============================================================================

# Qt6 dependencies - GUI needs full widget support
target_link_libraries(blackchirp-gui
    PUBLIC
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::Network
        QWT::QWT
        Blackchirp::Data
    PRIVATE
        GSL::gsl
        GSL::gslcblas
)

# Add GSL math library on Unix systems
if(UNIX)
    target_link_libraries(blackchirp-gui PRIVATE m)
endif()

# ============================================================================
# Compile Definitions
# ============================================================================

# Add version and configuration definitions
add_blackchirp_definitions(blackchirp-gui)

# GUI-layer specific definitions
target_compile_definitions(blackchirp-gui PRIVATE
    BC_GUI_LIBRARY
)

# ============================================================================
# Installation Configuration
# ============================================================================

# Install library
install(TARGETS blackchirp-gui
    EXPORT BlackchirpGuiTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Libraries
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Libraries
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Applications
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/blackchirp/gui
    COMPONENT Development
    FILES_MATCHING PATTERN "*.h"
)

# Export targets
install(EXPORT BlackchirpGuiTargets
    FILE BlackchirpGuiTargets.cmake
    NAMESPACE Blackchirp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Blackchirp
    COMPONENT Development
)

# ============================================================================
# Feature-Specific Configuration
# ============================================================================

# CUDA support (if enabled)
if(BC_ENABLE_CUDA)
    # GUI layer might use CUDA for visualization acceleration
    target_link_libraries(blackchirp-gui PRIVATE
        CUDA::cudart
    )
endif()

# Platform-specific configuration
if(WIN32)
    # Windows-specific GUI configuration
    target_compile_definitions(blackchirp-gui PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
elseif(APPLE)
    # macOS-specific GUI configuration
    target_link_libraries(blackchirp-gui PRIVATE
        ${COCOA_LIBRARY}
        ${IOKIT_LIBRARY}
        ${COREFOUNDATION_LIBRARY}
    )
elseif(UNIX)
    # Linux-specific GUI configuration
    if(UDEV_FOUND)
        target_link_libraries(blackchirp-gui PRIVATE ${UDEV_LIBRARIES})
        target_include_directories(blackchirp-gui PRIVATE ${UDEV_INCLUDE_DIRS})
    endif()
endif()

# ============================================================================
# Development and Debugging
# ============================================================================

# Add compile options for better debugging
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(blackchirp-gui PRIVATE
        BC_DEBUG
        QT_QML_DEBUG
    )
endif()

# Enable Qt logging for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(blackchirp-gui PRIVATE
        QT_MESSAGELOGCONTEXT
    )
endif()

# ============================================================================
# Status Information
# ============================================================================

message(STATUS "Blackchirp GUI Layer Configuration:")
message(STATUS "  Sources: ${BLACKCHIRP_GUI_SOURCES}")
message(STATUS "  Qt6 components: Core, Gui, Widgets, Network")
message(STATUS "  QWT integration: ${QWT_FOUND}")
message(STATUS "  GSL integration: ${GSL_FOUND}")
if(BC_ENABLE_CUDA)
    message(STATUS "  CUDA support: enabled")
endif()

# ============================================================================
# Internal Helper Functions
# ============================================================================

# Function to get GUI layer file lists (for use by other modules)
function(get_blackchirp_gui_files SOURCES_VAR HEADERS_VAR FORMS_VAR)
    set(${SOURCES_VAR} ${BLACKCHIRP_GUI_SOURCES} PARENT_SCOPE)
    set(${HEADERS_VAR} ${BLACKCHIRP_GUI_HEADERS} PARENT_SCOPE)
    set(${FORMS_VAR} ${BLACKCHIRP_GUI_FORMS} PARENT_SCOPE)
endfunction()

# Function to check if GUI layer has specific features
function(blackchirp_gui_has_feature FEATURE RESULT_VAR)
    set(${RESULT_VAR} FALSE PARENT_SCOPE)
    
    if(FEATURE STREQUAL "QWT")
        if(QWT_FOUND)
            set(${RESULT_VAR} TRUE PARENT_SCOPE)
        endif()
    elseif(FEATURE STREQUAL "CUDA")
        if(BC_ENABLE_CUDA)
            set(${RESULT_VAR} TRUE PARENT_SCOPE)
        endif()
    endif()
endfunction()