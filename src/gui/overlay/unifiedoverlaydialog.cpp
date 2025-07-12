#include "unifiedoverlaydialog.h"
#include "unifiedoverlaywidget.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDialogButtonBox>
#include <QLabel>
#include <QPushButton>
#include <QMessageBox>
#include <QDebug>

// Creation mode constructor
UnifiedOverlayDialog::UnifiedOverlayDialog(OverlayBase::OverlayType type,
                                         const QStringList &plotNames,
                                         double xRangeMin, double xRangeMax,
                                         const QVector<std::shared_ptr<OverlayBase>> &existingOverlays,
                                         QWidget *parent)
    : QDialog(parent),
      p_widget(nullptr),
      p_buttonBox(nullptr),
      p_statusLabel(nullptr),
      p_mainLayout(nullptr),
      d_mode(Mode::Creation),
      d_overlayType(type),
      d_isValid(false)
{
    setupUI();
    
    // Setup widget for creation mode
    p_widget->setupForCreation(type, plotNames, xRangeMin, xRangeMax, existingOverlays);
    
    setupConnections();
    updateWindowTitle();
    updateButtonState();
}

// Settings mode constructor
UnifiedOverlayDialog::UnifiedOverlayDialog(std::shared_ptr<OverlayBase> overlay,
                                         const QStringList &plotNames,
                                         double xRangeMin, double xRangeMax,
                                         std::shared_ptr<OverlayStorage> overlayStorage,
                                         QWidget *parent)
    : QDialog(parent),
      p_widget(nullptr),
      p_buttonBox(nullptr),
      p_statusLabel(nullptr),
      p_mainLayout(nullptr),
      d_mode(Mode::Settings),
      d_overlayType(overlay ? overlay->type() : OverlayBase::BCExperiment),
      d_overlay(overlay),
      d_isValid(true) // Settings mode starts valid
{
    setupUI();
    
    // Setup widget for settings mode
    p_widget->setupForSettings(overlay, plotNames, xRangeMin, xRangeMax, overlayStorage);
    
    setupConnections();
    updateWindowTitle();
    updateButtonState();
}

UnifiedOverlayDialog::~UnifiedOverlayDialog() = default;

std::shared_ptr<OverlayBase> UnifiedOverlayDialog::getOverlay() const
{
    return d_createdOverlay;
}

bool UnifiedOverlayDialog::isInPreviewMode() const
{
    return p_widget ? p_widget->isInPreviewMode() : false;
}

void UnifiedOverlayDialog::accept()
{
    if (!d_isValid) {
        QMessageBox::warning(this, "Validation Error", d_validationMessage);
        return;
    }
    
    if (isCreationMode()) {
        // Create the overlay from current settings
        d_createdOverlay = p_widget->createOverlay();
        if (!d_createdOverlay) {
            QMessageBox::critical(this, "Creation Error", "Failed to create overlay from current settings.");
            return;
        }
        
        // If in preview mode, clear preview flag before accepting
        if (p_widget->isInPreviewMode()) {
            d_createdOverlay->setPreview(false);
        }
        
    } else if (isSettingsMode()) {
        // Apply current settings to existing overlay
        p_widget->applyToOverlay();
        
        // If in preview mode, disable it before accepting
        if (p_widget->isInPreviewMode()) {
            p_widget->disablePreviewMode();
        }
    }
    
    QDialog::accept();
}

void UnifiedOverlayDialog::reject()
{
    // If in preview mode, disable it before rejecting
    if (p_widget && p_widget->isInPreviewMode()) {
        p_widget->disablePreviewMode();
    }
    
    QDialog::reject();
}

void UnifiedOverlayDialog::onValidationStatusChanged(bool isValid, const QString &message)
{
    d_isValid = isValid;
    d_validationMessage = message;
    
    updateButtonState();
    
    // Update status label
    if (p_statusLabel) {
        if (isValid) {
            p_statusLabel->setText("Settings are valid");
            p_statusLabel->setStyleSheet("QLabel { color: green; }");
        } else {
            p_statusLabel->setText(message.isEmpty() ? "Settings are invalid" : message);
            p_statusLabel->setStyleSheet("QLabel { color: red; }");
        }
    }
}

void UnifiedOverlayDialog::onPreviewRequested()
{
    // Forward signal and update button states
    emit previewRequested();
    updateButtonState();
    updateWindowTitle();
}

void UnifiedOverlayDialog::onPreviewCancelled()
{
    // Forward signal and update button states
    emit previewCancelled();
    updateButtonState();
    updateWindowTitle();
}

void UnifiedOverlayDialog::onOverlayDataChanged(std::shared_ptr<OverlayBase> overlay)
{
    // Forward real-time overlay updates
    emit overlayDataChanged(overlay);
}

void UnifiedOverlayDialog::setupUI()
{
    setModal(true);
    resize(800, 600);
    
    // Create main layout
    p_mainLayout = new QVBoxLayout(this);
    p_mainLayout->setContentsMargins(12, 12, 12, 12);
    p_mainLayout->setSpacing(12);
    
    // Create the unified overlay widget
    QString settingsKey = QString("UnifiedOverlayDialog_%1").arg(static_cast<int>(d_overlayType));
    p_widget = new UnifiedOverlayWidget(settingsKey, this);
    p_mainLayout->addWidget(p_widget, 1); // Give it all available space
    
    // Create status label
    p_statusLabel = new QLabel(this);
    p_statusLabel->setStyleSheet("QLabel { color: gray; font-style: italic; }");
    p_statusLabel->setText("Validating settings...");
    p_mainLayout->addWidget(p_statusLabel);
    
    // Create button box
    p_buttonBox = new QDialogButtonBox(this);
    if (isCreationMode()) {
        p_buttonBox->setStandardButtons(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
        p_buttonBox->button(QDialogButtonBox::Ok)->setText("Create Overlay");
    } else {
        p_buttonBox->setStandardButtons(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
        p_buttonBox->button(QDialogButtonBox::Ok)->setText("Apply Changes");
    }
    
    p_mainLayout->addWidget(p_buttonBox);
    
    // Connect button box
    connect(p_buttonBox, &QDialogButtonBox::accepted, this, &UnifiedOverlayDialog::accept);
    connect(p_buttonBox, &QDialogButtonBox::rejected, this, &UnifiedOverlayDialog::reject);
}

void UnifiedOverlayDialog::setupConnections()
{
    if (!p_widget) {
        return;
    }
    
    // Validation status
    connect(p_widget, &UnifiedOverlayWidget::validationStatusChanged,
            this, &UnifiedOverlayDialog::onValidationStatusChanged);
    
    // Preview mode signals
    connect(p_widget, &UnifiedOverlayWidget::previewRequested,
            this, &UnifiedOverlayDialog::onPreviewRequested);
    connect(p_widget, &UnifiedOverlayWidget::previewCancelled,
            this, &UnifiedOverlayDialog::onPreviewCancelled);
    
    // Real-time overlay updates (settings mode only)
    if (isSettingsMode()) {
        connect(p_widget, &UnifiedOverlayWidget::overlayDataChanged,
                this, &UnifiedOverlayDialog::onOverlayDataChanged);
    }
}

void UnifiedOverlayDialog::updateButtonState()
{
    if (!p_buttonBox) {
        return;
    }
    
    QPushButton *okButton = p_buttonBox->button(QDialogButtonBox::Ok);
    if (okButton) {
        // In creation mode, OK is only enabled if settings are valid
        // In settings mode, OK is always enabled (can apply partial changes)
        bool enableOk = isSettingsMode() || d_isValid;
        okButton->setEnabled(enableOk);
        
        // Update button text based on preview mode
        if (isInPreviewMode()) {
            okButton->setText(isCreationMode() ? "Create from Preview" : "Apply from Preview");
        } else {
            okButton->setText(isCreationMode() ? "Create Overlay" : "Apply Changes");
        }
    }
}

void UnifiedOverlayDialog::updateWindowTitle()
{
    QString contextName = getContextName();
    QString typeName = getTypeName();
    QString previewSuffix = isInPreviewMode() ? " (Preview)" : "";
    
    setWindowTitle(QString("%1 %2 Overlay%3").arg(contextName, typeName, previewSuffix));
}

QString UnifiedOverlayDialog::getContextName() const
{
    switch (d_mode) {
    case Mode::Creation:
        return "Create";
    case Mode::Settings:
        return "Configure";
    }
    return "Overlay";
}

QString UnifiedOverlayDialog::getTypeName() const
{
    switch (d_overlayType) {
    case OverlayBase::BCExperiment:
        return "BC Experiment";
    case OverlayBase::Catalog:
        return "Catalog";
    case OverlayBase::GenericXY:
        return "Generic XY";
    }
    return "Unknown";
}

bool UnifiedOverlayDialog::isCreationMode() const
{
    return d_mode == Mode::Creation;
}

bool UnifiedOverlayDialog::isSettingsMode() const
{
    return d_mode == Mode::Settings;
}