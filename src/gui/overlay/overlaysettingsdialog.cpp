#include "overlaysettingsdialog.h"
#include <QMessageBox>
#include <QPushButton>
#include <QGroupBox>

OverlaySettingsDialog::OverlaySettingsDialog(std::shared_ptr<OverlayBase> overlay, 
                                           const QStringList &plotNames,
                                           double xRangeMin, double xRangeMax, 
                                           QWidget *parent)
    : QDialog(parent),
      d_overlay(overlay),
      d_plotNames(plotNames),
      d_xRangeMin(xRangeMin),
      d_xRangeMax(xRangeMax),
      p_mainLayout(nullptr),
      p_optionsWidget(nullptr),
      p_buttonBox(nullptr),
      p_resetButton(nullptr),
      p_titleLabel(nullptr)
{
    if (!d_overlay) {
        reject();
        return;
    }

    // Note: setupUI() will be called after construction by the creator
}

OverlaySettingsDialog::~OverlaySettingsDialog()
{
}

void OverlaySettingsDialog::setupUI()
{
    setWindowTitle("Configure Overlay Settings");
    setModal(true);
    resize(450, 400);

    p_mainLayout = new QVBoxLayout(this);

    // Title label showing overlay name
    p_titleLabel = new QLabel(QString("Configure: %1").arg(d_overlay->getLabel()), this);
    p_titleLabel->setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;");
    p_mainLayout->addWidget(p_titleLabel);

    // Create options widget for base overlay settings
    QGroupBox *baseGroup = new QGroupBox("Overlay Settings", this);
    QVBoxLayout *baseLayout = new QVBoxLayout(baseGroup);
    
    // Create options widget with plot names from parent
    p_optionsWidget = new OverlayBaseOptionsWidget(d_plotNames, d_xRangeMin, d_xRangeMax, this);
    baseLayout->addWidget(p_optionsWidget);
    
    p_mainLayout->addWidget(baseGroup);

    // Call virtual function for type-specific UI setup
    setupTypeSpecificUI();

    // Add stretch to push buttons to bottom
    p_mainLayout->addStretch();

    // Create button box with custom Reset button
    p_buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    p_resetButton = new QPushButton("Reset to Defaults", this);
    p_buttonBox->addButton(p_resetButton, QDialogButtonBox::ResetRole);
    
    p_mainLayout->addWidget(p_buttonBox);

    setLayout(p_mainLayout);

    // Set up connections after all UI elements exist
    setupConnections();
    
    // Load current settings
    loadCurrentSettings();
}

void OverlaySettingsDialog::setupConnections()
{
    // Connect dialog buttons
    connect(p_buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(p_buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(p_resetButton, &QPushButton::clicked, this, &OverlaySettingsDialog::onResetToDefaults);

    // For now, we'll apply settings when the dialog is accepted
    // Future enhancement: add real-time updates by connecting to individual widget signals
    // The existing architecture already supports real-time updates through onSettingsChanged()

    // Call virtual function for type-specific connections
    setupTypeSpecificConnections();
}

void OverlaySettingsDialog::loadCurrentSettings()
{
    // Store original values for reset functionality
    d_originalLabel = d_overlay->getLabel();
    d_originalPlotId = d_overlay->getPlotId();
    d_originalYScale = d_overlay->getYScale();
    d_originalYOffset = d_overlay->getYOffset();
    d_originalXOffset = d_overlay->getXOffset();
    d_originalMinFreqEnabled = d_overlay->getMinFreqEnabled();
    d_originalMinFreqValue = d_overlay->getMinFreqValue();
    d_originalMaxFreqEnabled = d_overlay->getMaxFreqEnabled();
    d_originalMaxFreqValue = d_overlay->getMaxFreqValue();

    // Load current settings into the options widget
    p_optionsWidget->setLabel(d_originalLabel);
    p_optionsWidget->setPlotId(d_originalPlotId);
    p_optionsWidget->setYScale(d_originalYScale);
    p_optionsWidget->setYOffset(d_originalYOffset);
    p_optionsWidget->setXOffset(d_originalXOffset);
    p_optionsWidget->setMinFreqLimit(d_originalMinFreqEnabled, d_originalMinFreqValue);
    p_optionsWidget->setMaxFreqLimit(d_originalMaxFreqEnabled, d_originalMaxFreqValue);

    // Call virtual function for type-specific loading
    loadTypeSpecificSettings();
}

void OverlaySettingsDialog::saveCurrentSettings()
{
    // Apply settings from options widget to overlay using the existing method
    p_optionsWidget->applyToOverlay(d_overlay);

    // Call virtual function for type-specific saving
    saveTypeSpecificSettings();
}

void OverlaySettingsDialog::onSettingsChanged()
{
    // Save current settings to overlay
    saveCurrentSettings();
    
    // Emit signal for real-time updates
    emit overlaySettingsChanged(d_overlay);
}

void OverlaySettingsDialog::onResetToDefaults()
{
    // Reset to original values
    p_optionsWidget->setLabel(d_originalLabel);
    p_optionsWidget->setPlotId(d_originalPlotId);
    p_optionsWidget->setYScale(d_originalYScale);
    p_optionsWidget->setYOffset(d_originalYOffset);
    p_optionsWidget->setXOffset(d_originalXOffset);
    p_optionsWidget->setMinFreqLimit(d_originalMinFreqEnabled, d_originalMinFreqValue);
    p_optionsWidget->setMaxFreqLimit(d_originalMaxFreqEnabled, d_originalMaxFreqValue);

    // Call virtual function for type-specific reset
    resetTypeSpecificDefaults();
    
    // Trigger settings changed to apply and update UI
    onSettingsChanged();
}

void OverlaySettingsDialog::accept()
{
    // Save current settings when dialog is accepted
    saveCurrentSettings();
    
    // Emit final signal for any updates
    emit overlaySettingsChanged(d_overlay);
    
    // Call base class accept
    QDialog::accept();
}