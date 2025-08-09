#include "runtimehardwareconfigdialog.h"
#include "runtimehardwareconfigdialog_ui.h"

RuntimeHardwareConfigDialog::RuntimeHardwareConfigDialog(QWidget *parent)
    : QDialog(parent),
      pu_ui(new Ui::RuntimeHardwareConfigDialog)
{
    pu_ui->setupUi(this);
    
    // Apply theme-aware styling to validation status label
    pu_ui->applyValidationStatusStyling(this);
    
    // Connect dialog buttons
    connect(pu_ui->buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(pu_ui->buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

RuntimeHardwareConfigDialog::~RuntimeHardwareConfigDialog()
{
    delete pu_ui;
}