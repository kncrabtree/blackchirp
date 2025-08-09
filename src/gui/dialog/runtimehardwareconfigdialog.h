#ifndef RUNTIMEHARDWARECONFIGDIALOG_H
#define RUNTIMEHARDWARECONFIGDIALOG_H

#include <QDialog>
#include <QTreeWidgetItem>
#include <QStringList>
#include <QPair>
#include <hardware/core/runtimehardwareconfig.h>
#include <data/bcglobals.h>

// Forward declarations
class QListWidgetItem;

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
    

    Ui::RuntimeHardwareConfigDialog *pu_ui;
};

#endif // RUNTIMEHARDWARECONFIGDIALOG_H