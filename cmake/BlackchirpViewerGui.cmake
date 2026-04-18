# BlackchirpViewerGui.cmake - Viewer-specific GUI components for Blackchirp
#
# This module defines the blackchirp-viewer-gui library target containing:
# - Experiment viewing widgets  
# - Plotting components and QWT integration
# - Data visualization components 
# - Theme and styling components
# - Data models for GUI components
# 
# Excludes main application components that depend on hardware layer

# Include guard to prevent multiple inclusions
if(BLACKCHIRP_VIEWER_GUI_CMAKE_INCLUDED)
    return()
endif()
set(BLACKCHIRP_VIEWER_GUI_CMAKE_INCLUDED TRUE)

# ============================================================================
# Viewer GUI Layer Source Files
# ============================================================================

set(BLACKCHIRP_VIEWER_GUI_SOURCES
    # Experiment viewing widgets (viewer-specific)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/experimentviewwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwviewwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/auxdataviewwidget.cpp
    
    # Plotting components (safe for viewer)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/blackchirpplotcurve.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/curveappearancewidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/curveappearancepresetmanager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/presetsavedialog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/curvefactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/customtracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/customzoomer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/fidplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/ftplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/mainftplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/trackingplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/zoompanplot.cpp
    
    # Overlay system (viewer uses overlays for data visualization)
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
    
    # Dialogs (viewer-safe)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/peaklistexportdialog.cpp
    
    # Basic widgets (no hardware dependencies)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/experimentsummarywidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwplottoolbar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwprocessingtoolbar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/led.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/peakfindwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/scientificspinbox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/scientificinputwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/util/numericformat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/toolbarwidgetaction.cpp
    
    # Styling and theming
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/style/themecolors.cpp
    
    # Data models (moved from data layer - they belong in GUI)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/exptsummarymodel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/overlaytablemodel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/peaklistmodel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/validationmodel.cpp
    
    # LIF GUI components (needed for viewing LIF experiment data)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifdisplaywidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifprocessingwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifsliceplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifspectrogramplot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/liftraceplot.cpp
)

set(BLACKCHIRP_VIEWER_GUI_HEADERS
    # Experiment viewing widgets (viewer-specific)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/experimentviewwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwviewwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/auxdataviewwidget.h
    
    # Plotting components (safe for viewer)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/blackchirpplotcurve.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/curveappearancewidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/curveappearancepresetmanager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/presetsavedialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/curvefactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/customtracker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/customzoomer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/fidplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/ftplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/mainftplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/trackingplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/plot/zoompanplot.h
    
    # Overlay system (viewer uses overlays for data visualization)
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
    
    # Dialogs (viewer-safe)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/peaklistexportdialog.h
    
    # Basic widgets (no hardware dependencies)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/enumcombobox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/experimentsummarywidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwplottoolbar.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/ftmwprocessingtoolbar.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/led.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/peakfindwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/scientificspinbox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/scientificinputwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/util/numericformat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/toolbarwidgetaction.h
    
    # Styling and theming
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/style/themecolors.h
    
    # Data models (moved from data layer - they belong in GUI)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/exptsummarymodel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/overlaytablemodel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/peaklistmodel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/model/validationmodel.h
    
    # LIF GUI components (needed for viewing LIF experiment data)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifdisplaywidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifprocessingwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifsliceplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/lifspectrogramplot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/lif/gui/liftraceplot.h
)

# UI Forms (only the ones safe for viewer)
set(BLACKCHIRP_VIEWER_GUI_FORMS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/dialog/peaklistexportdialog.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/widget/peakfindwidget.ui
)

# ============================================================================
# Create Viewer GUI Library Target
# ============================================================================

# Create the blackchirp-viewer-gui library
add_library(blackchirp-viewer-gui STATIC
    ${BLACKCHIRP_VIEWER_GUI_SOURCES}
    ${BLACKCHIRP_VIEWER_GUI_HEADERS}
    ${BLACKCHIRP_VIEWER_GUI_FORMS}
)

# Add alias for consistent naming
add_library(Blackchirp::ViewerGui ALIAS blackchirp-viewer-gui)

# ============================================================================
# Target Properties and Configuration
# ============================================================================

# Set target properties
set_target_properties(blackchirp-viewer-gui PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "blackchirp-viewer-gui"
    EXPORT_NAME "ViewerGui"
)

# Include directories
target_include_directories(blackchirp-viewer-gui
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
target_link_libraries(blackchirp-viewer-gui
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
    target_link_libraries(blackchirp-viewer-gui PRIVATE m)
endif()

# ============================================================================
# Compile Definitions
# ============================================================================

# Add version and configuration definitions
add_blackchirp_definitions(blackchirp-viewer-gui)

# Viewer GUI-layer specific definitions
target_compile_definitions(blackchirp-viewer-gui PRIVATE
    BC_VIEWER_GUI_LIBRARY
    BC_VIEWER
)

# ============================================================================
# Installation Configuration
# ============================================================================

# Install library
install(TARGETS blackchirp-viewer-gui
    EXPORT BlackchirpViewerGuiTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Libraries
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Libraries
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Applications
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(EXPORT BlackchirpViewerGuiTargets
    FILE BlackchirpViewerGuiTargets.cmake
    NAMESPACE Blackchirp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Blackchirp
    COMPONENT Development
)

# ============================================================================
# Status Information
# ============================================================================

message(STATUS "Blackchirp Viewer GUI Layer Configuration:")
message(STATUS "  Components: experiment viewing, plotting, overlays, basic widgets")
message(STATUS "  Qt6 components: Core, Gui, Widgets, Network")
message(STATUS "  QWT integration: ${QWT_FOUND}")
message(STATUS "  Hardware dependencies: EXCLUDED")