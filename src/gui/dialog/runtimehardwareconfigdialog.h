#ifndef RUNTIMEHARDWARECONFIGDIALOG_H
#define RUNTIMEHARDWARECONFIGDIALOG_H

#include <QDialog>
#include <QTreeWidgetItem>
#include <QStringList>
#include <QPair>
#include <hardware/core/runtimehardwareconfig.h>
#include <data/bcglobals.h>

// Forward declare UI class
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
    // Future implementation: add slots for hardware selection, validation, etc.
    
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
    

    Ui::RuntimeHardwareConfigDialog *pu_ui;
};

#endif // RUNTIMEHARDWARECONFIGDIALOG_H