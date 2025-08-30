#ifndef RUNTIMEHARDWARECONFIGDIALOG_H
#define RUNTIMEHARDWARECONFIGDIALOG_H

#include <QDialog>
#include <QTreeWidgetItem>
#include <QStringList>
#include <QPair>
#include <QMap>
#include <hardware/core/runtimehardwareconfig.h>
#include <data/bcglobals.h>

// Forward declarations
class QListWidgetItem;
class QComboBox;
class QLineEdit;
class QListWidget;
class QPushButton;
class QButtonGroup;
class QCheckBox;
class QRadioButton;
class QTableWidgetItem;
class VendorLibrary;
class SpectrumLibrary;
class LabjackLibrary;

namespace Ui {
class RuntimeHardwareConfigDialog;
}

/**
 * @brief Runtime Hardware Configuration Dialog
 * 
 * Provides user interface for selecting and configuring hardware implementations
 * at runtime. This minimal implementation establishes the framework for future
 * development of hardware configuration and profile management capabilities.
 */
class RuntimeHardwareConfigDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit RuntimeHardwareConfigDialog(QWidget *parent = nullptr);
    ~RuntimeHardwareConfigDialog();
    
private slots:
    /*!
     * \brief Handle hardware browser selection changes
     * \param current Currently selected item
     * \param previous Previously selected item (unused)
     */
    void onHardwareBrowserSelectionChanged(QListWidgetItem* current, QListWidgetItem* previous);
    
private:
    /*!
     * \brief Populate configuration overview tree with actual hardware configuration
     * 
     * Reads current hardware configuration from RuntimeHardwareConfig and populates
     * the left panel tree with properly formatted hardware display
     */
    void populateConfigurationOverview();
    
    /*!
     * \brief Clear and repopulate the configuration overview tree
     * 
     * Helper method to refresh the tree display after configuration changes
     */
    void refreshConfigurationOverview();
    
    /*!
     * \brief Populate hardware browser with available hardware types and counts
     * 
     * Connects middle panel QListWidget to HardwareRegistry::getHardwareTypes()
     * and displays format: "HardwareType (count)" where count shows active instances
     * from RuntimeHardwareConfig::getCurrentHardware() data
     */
    void populateHardwareBrowser();
    
    /*!
     * \brief Update right panel display based on hardware selection
     * \param selectedHardwareType Hardware type that was selected (e.g., "Clock")
     */
    void updateSelectionDisplay(const QString& selectedHardwareType);
    
    /*!
     * \brief Create and update the right panel UI for the specified hardware type
     * \param hardwareType Hardware type to create UI for (e.g., "Clock", "FtmwScope")
     * 
     * Uses HardwareRegistry::isMultiInstanceType() to determine whether to show
     * single-instance UI or multi-instance UI layout
     */
    void updateRightPanelForHardwareType(const QString& hardwareType);
    
    /*!
     * \brief Handle profile selection changes and update preview state
     * \param hardwareType Hardware type being configured
     */
    void onProfileSelectionChanged(const QString& hardwareType);
    
    /*!
     * \brief Update preview state and refresh left panel display
     */
    void updatePreviewConfiguration();
    
    /*!
     * \brief Handle Add Profile button click
     * \param hardwareType Hardware type to add profile for
     */
    void onAddProfile(const QString& hardwareType);
    
    /*!
     * \brief Handle Remove Profile button click
     * \param hardwareType Hardware type to remove profile from
     */
    void onRemoveProfile(const QString& hardwareType);
    
    /*!
     * \brief Handle dialog accept - apply preview configuration to RuntimeHardwareConfig
     */
    void onDialogAccepted();
    
    /*!
     * \brief Handle dialog cancel - restore original runtime configuration
     */
    void onDialogRejected();
    
    /*!
     * \brief Validate current preview configuration and update status bar
     * 
     * Updates the validation status bar with real-time feedback based on 
     * RuntimeHardwareConfig::validateConfiguration() results and controls
     * Apply button enablement based on validation state.
     */
    void validatePreviewConfiguration();
    
    /*!
     * \brief Update validation status bar with specified message and state
     * \param message Status message to display
     * \param state Validation state (Success/Error/Info)
     */
    void updateValidationStatus(const QString& message, const QString& state);
    
    // Phase 3.5: Library Status Tab Methods
    /*!
     * \brief Initialize the Library Status tab with vendor library information
     */
    void initializeLibraryStatusTab();
    
    /*!
     * \brief Refresh library status display with current information
     */
    void refreshLibraryStatus();
    
    /*!
     * \brief Handle library selection changes in the overview table
     * \param current Currently selected item
     * \param previous Previously selected item (unused)
     */
    void onLibrarySelectionChanged(QTableWidgetItem* current, QTableWidgetItem* previous);
    
    /*!
     * \brief Handle changes to library path configuration
     * \param libraryKey Settings key for the library being configured
     */
    void onLibraryPathChanged(const QString& libraryKey);
    
    /*!
     * \brief Handle Browse button click for library path selection
     */
    void onBrowseLibraryPath();
    
    /*!
     * \brief Handle Test Load button click to validate library configuration
     */
    void onTestLoadLibrary();
    
    /*!
     * \brief Update library details panel for selected library
     * \param library Reference to the selected vendor library
     */
    void updateLibraryDetails(VendorLibrary& library);
    
    /*!
     * \brief Update library configuration panel for selected library
     * \param library Reference to the selected vendor library
     */
    void updateLibraryConfiguration(VendorLibrary& library);
    
    /*!
     * \brief Get status text for library based on its availability
     * \param library Reference to the vendor library
     * \return Status string ("Available", "Not Found", "Error")
     */
    QString getLibraryStatusText(VendorLibrary& library);
    
    /*!
     * \brief Get display name for vendor library
     * \param library Reference to the vendor library
     * \return Human-readable library name
     */
    QString getLibraryDisplayName(VendorLibrary& library);
    
    /*!
     * \brief Get version information for library if available
     * \param library Reference to the vendor library
     * \return Version string or "Unknown" if not available
     */
    QString getLibraryVersion(VendorLibrary& library);
    
    /*!
     * \brief Update visual indicators for staged library changes
     * 
     * Shows visual feedback (asterisks, colors, etc.) when library
     * configuration has unstaged changes pending application.
     */
    void updateStagingIndicators();
    
    /*!
     * \brief Update visual indication for individual UI control staging state
     * \param control UI control to update (QLineEdit, QCheckBox, etc.)
     * \param isModified Whether the control has unstaged changes
     */
    void updateControlStagingIndicator(QWidget* control, bool isModified);
    
    /*!
     * \brief Revert staged changes for all vendor libraries
     * 
     * Called when dialog is cancelled to discard all pending library changes.
     */
    void revertAllLibraryChanges();
    
    /*!
     * \brief Update staging indicators for all libraries including tab-level indicators
     * 
     * Updates the Library Status tab title and other global indicators when
     * any library has unstaged changes.
     */
    void updateAllLibraryStagingIndicators();

    Ui::RuntimeHardwareConfigDialog *pu_ui;
    
    // State management for Phase 4.3
    std::map<QString, QString> d_originalRuntimeConfig;  // Original runtime configuration for cancel functionality
    std::map<QString, QString> d_previewRuntimeConfig;   // Preview of runtime configuration changes
    QString d_currentHardwareType;  // Currently selected hardware type in browser
    
    // Phase 3.5: Library Status Tab state
    QString d_currentLibraryKey;      // Currently selected library key
    VendorLibrary* p_currentLibrary;  // Pointer to currently selected library
};

#endif // RUNTIMEHARDWARECONFIGDIALOG_H