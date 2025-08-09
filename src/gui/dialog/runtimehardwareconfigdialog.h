#ifndef RUNTIMEHARDWARECONFIGDIALOG_H
#define RUNTIMEHARDWARECONFIGDIALOG_H

#include <QDialog>

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
    Ui::RuntimeHardwareConfigDialog *pu_ui;
};

#endif // RUNTIMEHARDWARECONFIGDIALOG_H