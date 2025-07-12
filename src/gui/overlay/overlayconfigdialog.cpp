#include "overlayconfigdialog.h"
#include <QMessageBox>
#include <QVBoxLayout>
#include <gui/widget/ftmwviewwidget.h>

OverlayConfigDialog::OverlayConfigDialog(FtmwViewWidget *parent)
    : QDialog(parent),
      p_overlayOptionsWidget(nullptr),
      p_buttonBox(nullptr),
      p_validationLabel(nullptr)
{
    // Access parameters directly from parent FtmwViewWidget
    if (parent) {
        d_plotNames = parent->getPlotNames();
        auto xRange = parent->getMainPlotFt().xRange();
        d_xRangeMin = xRange.first;
        d_xRangeMax = xRange.second;
        d_existingOverlays = parent->getAllOverlays();
    }
    
    // Note: setupUI() will be called after construction by the creator
}

void OverlayConfigDialog::setupUI()
{
    setModal(true);
    resize(450, 400);
    
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // Set up overlay base options first
    setupOverlayBaseOptions();
    
    // Set up type-specific UI (derived classes can now safely be called)
    setupTypeSpecificUI();
    
    // Add validation label
    p_validationLabel = new QLabel(this);
    p_validationLabel->setStyleSheet("QLabel { color: red; }");
    p_validationLabel->setWordWrap(true);
    p_validationLabel->hide();
    mainLayout->addWidget(p_validationLabel);
    
    // Add button box at the bottom
    p_buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    p_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
    mainLayout->addWidget(p_buttonBox);
    
    setLayout(mainLayout);
    
    // Set up connections after all UI elements exist
    setupCommonConnections();
    setupTypeSpecificConnections();
    
    // Initialize defaults
    initializeCommonDefaults();
    initializeTypeSpecificDefaults();
}

void OverlayConfigDialog::setupOverlayBaseOptions()
{
    QGroupBox *optionsGroup = new QGroupBox("Overlay Options", this);
    QVBoxLayout *optionsLayout = new QVBoxLayout(optionsGroup);
    
    // Create the options widget with the plot names and xRange
    p_overlayOptionsWidget = new OverlayBaseOptionsWidget(d_plotNames, d_xRangeMin, d_xRangeMax, this);
    optionsLayout->addWidget(p_overlayOptionsWidget);
    
    layout()->addWidget(optionsGroup);
}

void OverlayConfigDialog::setupCommonConnections()
{
    // Dialog buttons
    connect(p_buttonBox, &QDialogButtonBox::accepted,
            this, &OverlayConfigDialog::onDialogAccepted);
    connect(p_buttonBox, &QDialogButtonBox::rejected,
            this, &QDialog::reject);
}

void OverlayConfigDialog::initializeCommonDefaults()
{
    // Base options widget will initialize its own defaults
    updateOkButtonState();
}

void OverlayConfigDialog::updateValidationStatus(bool valid, const QString &message)
{
    if (valid) {
        p_validationLabel->hide();
    } else {
        p_validationLabel->setText(message);
        p_validationLabel->show();
    }
}

void OverlayConfigDialog::updateOkButtonState()
{
    bool canAccept = isTypeSpecificDataValid();
    p_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(canAccept);
}

void OverlayConfigDialog::onDialogAccepted()
{
    // Additional validation could go here
    accept();
}

void OverlayConfigDialog::accept()
{
    QStringList validationErrors;
    
    // Validate overlay base options (including overlay existence check)
    if (p_overlayOptionsWidget) {
        QString overlayError;
        if (!p_overlayOptionsWidget->validateSettings(overlayError, d_existingOverlays)) {
            validationErrors << overlayError;
        }
    }
    
    // Validate type-specific settings
    QString typeSpecificError;
    if (!validateTypeSpecificSettings(typeSpecificError)) {
        validationErrors << typeSpecificError;
    }
    
    // Show validation errors if any
    if (!validationErrors.isEmpty()) {
        QString errorMessage = "Please fix the following issues:\n\n";
        errorMessage += validationErrors.join("\n");
        
        QMessageBox::warning(this, "Validation Error", errorMessage);
        return; // Don't close dialog
    }
    
    // All validation passed, call base class accept
    QDialog::accept();
}

std::shared_ptr<OverlayBase> OverlayConfigDialog::createOverlay() const
{
    // Template Method pattern implementation
    
    // 1. Validate that we can create an overlay
    if (!isTypeSpecificDataValid()) {
        return nullptr;
    }
    
    // 2. Create the type-specific overlay (delegated to derived class)
    auto overlay = createTypeSpecificOverlay();
    if (!overlay) {
        return nullptr;
    }
    
    // 3. Apply common base options (handled by base class)
    if (p_overlayOptionsWidget) {
        p_overlayOptionsWidget->applyToOverlay(overlay);
    }
    
    // 4. Allow derived class to perform additional configuration
    configureTypeSpecificOverlay(overlay);
    
    return overlay;
}
