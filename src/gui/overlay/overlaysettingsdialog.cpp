#include "overlaysettingsdialog.h"
#include <QMessageBox>
#include <QPushButton>
#include <QGroupBox>
#include <QCloseEvent>
#include <QColorDialog>
#include <gui/plot/blackchirpplotcurve.h>

OverlaySettingsDialog::OverlaySettingsDialog(std::shared_ptr<OverlayBase> overlay, 
                                           const QStringList &plotNames,
                                           double xRangeMin, double xRangeMax,
                                           std::shared_ptr<OverlayStorage> overlayStorage,
                                           QWidget *parent)
    : QDialog(parent),
      SettingsStorage(BC::Key::OverlaySettings::key),
      d_overlay(overlay),
      d_plotNames(plotNames),
      d_xRangeMin(xRangeMin),
      d_xRangeMax(xRangeMax),
      p_overlayStorage(overlayStorage),
      p_mainLayout(nullptr),
      p_optionsWidget(nullptr),
      p_curveAppearanceWidget(nullptr),
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

    // Create curve appearance section
    QGroupBox *curveGroup = new QGroupBox("Curve Appearance", this);
    QVBoxLayout *curveLayout = new QVBoxLayout(curveGroup);
    
    p_curveAppearanceWidget = new CurveAppearanceWidget(this);
    curveLayout->addWidget(p_curveAppearanceWidget);
    
    p_mainLayout->addWidget(curveGroup);

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

    // Load current settings before establishing connections
    loadCurrentSettings();
    
    // Set up connections after all UI elements exist and are initialized
    setupConnections();
    
    // Restore dialog geometry if available
    QByteArray geom = get(BC::Key::OverlaySettings::geometry).toByteArray();
    if (!geom.isEmpty()) {
        restoreGeometry(geom);
    }
}

void OverlaySettingsDialog::setupConnections()
{
    // Connect dialog buttons
    connect(p_buttonBox, &QDialogButtonBox::accepted, this, &OverlaySettingsDialog::accept);
    connect(p_buttonBox, &QDialogButtonBox::rejected, this, &OverlaySettingsDialog::reject);
    connect(p_resetButton, &QPushButton::clicked, this, &OverlaySettingsDialog::onResetToDefaults);

    // Connect to options widget's settingsChanged signal for real-time updates
    connect(p_optionsWidget, &OverlayBaseOptionsWidget::settingsChanged,
            this, &OverlaySettingsDialog::onRealTimeSettingsChanged);
    
    // Connect to curve appearance widget for real-time curve updates
    connect(p_curveAppearanceWidget, &CurveAppearanceWidget::curveAppearanceChanged,
            this, &OverlaySettingsDialog::onRealTimeSettingsChanged);
    
    // Connect color change requests to handle color dialog
    connect(p_curveAppearanceWidget, &CurveAppearanceWidget::colorChangeRequested,
            this, &OverlaySettingsDialog::onColorChangeRequested);

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

    // Initialize curve appearance widget from overlay metadata
    p_curveAppearanceWidget->initializeFromOverlay(d_overlay);
    
    // Store original curve appearance for reset/cancel functionality
    d_originalCurveAppearance = p_curveAppearanceWidget->getCurrentAppearance();

    // Call virtual function for type-specific loading
    loadTypeSpecificSettings();
}

void OverlaySettingsDialog::saveCurrentSettings()
{
    // Apply settings from options widget to overlay using the existing method
    p_optionsWidget->applyToOverlay(d_overlay);
    
    // Save curve appearance settings to overlay metadata
    p_curveAppearanceWidget->applyToOverlay(d_overlay);

    // Call virtual function for type-specific saving
    saveTypeSpecificSettings();
}

void OverlaySettingsDialog::onRealTimeSettingsChanged()
{
    // Apply settings from options widget to overlay, but preserve original label for real-time updates
    // The label will only be changed when the dialog is accepted to trigger file renaming
    QString currentLabel = d_overlay->getLabel();
    
    // Apply all settings including temporary label
    p_optionsWidget->applyToOverlay(d_overlay);
    
    // Restore original label to prevent file renaming during real-time updates
    d_overlay->setLabel(currentLabel);

    // Save curve appearance settings to overlay metadata for real-time updates
    p_curveAppearanceWidget->applyToOverlay(d_overlay);

    // Call virtual function for type-specific saving (for real-time updates)
    saveTypeSpecificSettings();
    
    // Emit signal for real-time plot updates
    emit overlaySettingsChanged(d_overlay);
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

    // Reset curve appearance to original values
    p_curveAppearanceWidget->setCurrentAppearance(d_originalCurveAppearance);

    // Call virtual function for type-specific reset
    resetTypeSpecificDefaults();
    
    // Trigger settings changed to apply and update UI
    onSettingsChanged();
}

void OverlaySettingsDialog::accept()
{
    // Check if label has changed and handle renaming if needed
    QString newLabel = p_optionsWidget->getLabel();
    if (p_overlayStorage && newLabel != d_originalLabel) {
        if (!p_overlayStorage->renameOverlay(d_originalLabel, newLabel)) {
            QMessageBox::warning(this, "Rename Failed", 
                                QString("Failed to rename overlay from '%1' to '%2'. "
                                       "Please check that the new name is valid and not already in use.")
                                       .arg(d_originalLabel, newLabel));
            return; // Don't close dialog on rename failure
        }
    }
    
    // Save current settings when dialog is accepted
    saveCurrentSettings();
    
    // Emit final signal for any updates
    emit overlaySettingsChanged(d_overlay);
    
    // Save dialog geometry
    set(BC::Key::OverlaySettings::geometry, saveGeometry(), true);
    
    // Call base class accept
    QDialog::accept();
}

void OverlaySettingsDialog::reject()
{
    // Restore original curve appearance to undo any real-time changes
    p_curveAppearanceWidget->setCurrentAppearance(d_originalCurveAppearance);
    p_curveAppearanceWidget->applyToOverlay(d_overlay);
    
    // Emit signal to update plots with restored appearance
    emit overlaySettingsChanged(d_overlay);
    
    // Save dialog geometry
    set(BC::Key::OverlaySettings::geometry, saveGeometry(), true);
    
    // Call base class reject
    QDialog::reject();
}

void OverlaySettingsDialog::closeEvent(QCloseEvent *event)
{
    // Save dialog geometry
    set(BC::Key::OverlaySettings::geometry, saveGeometry(), true);
    
    // Accept the close event
    QDialog::closeEvent(event);
}


void OverlaySettingsDialog::onColorChangeRequested()
{
    if (!p_curveAppearanceWidget) {
        return;
    }
    
    // Get current color from the widget
    QColor currentColor = p_curveAppearanceWidget->getCurrentAppearance().color;
    
    // Open color dialog
    QColor newColor = QColorDialog::getColor(currentColor, this, "Choose Curve Color");
    
    // Update the widget if a valid color was chosen
    if (newColor.isValid()) {
        p_curveAppearanceWidget->updateColorDisplay(newColor);
    }
}