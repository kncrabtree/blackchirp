#include "unifiedoverlaydialog.h"
#include "unifiedoverlaywidget.h"
#include "overlaytypespecificwidget.h"
#include <data/processing/overlayoperation.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDialogButtonBox>
#include <QLabel>
#include <QPushButton>
#include <QProgressBar>
#include <QMessageBox>
#include <QTimer>
#include <QApplication>
#include <QDebug>
#include <gui/plot/blackchirpplotcurve.h>

// Creation mode constructor
UnifiedOverlayDialog::UnifiedOverlayDialog(OverlayBase::OverlayType type,
                                         const QStringList &plotNames,
                                         const Ft &currentFt,
                                         const QVector<std::shared_ptr<OverlayBase>> &existingOverlays,
                                         QWidget *parent)
    : QDialog(parent),
      p_widget(nullptr),
      p_buttonBox(nullptr),
      p_statusLabel(nullptr),
      p_mainLayout(nullptr),
      p_progressBar(nullptr),
      p_progressLabel(nullptr),
      p_cancelButton(nullptr),
      p_progressTimer(nullptr),
      p_timeoutTimer(nullptr),
      d_mode(Mode::Creation),
      d_overlayType(type),
      d_dialogState(DialogState::Ready),
      d_operationProgress(0),
      d_isValid(false)
{
    setupUI();
    
    // Setup widget for creation mode
    p_widget->setupForCreation(type, plotNames, currentFt, existingOverlays);
    
    setupConnections();
    updateWindowTitle();
    updateButtonState();
}

// Settings mode constructor
UnifiedOverlayDialog::UnifiedOverlayDialog(std::shared_ptr<OverlayBase> overlay,
                                         const QStringList &plotNames,
                                         const Ft &currentFt,
                                         std::shared_ptr<OverlayStorage> overlayStorage,
                                         QWidget *parent)
    : QDialog(parent),
      p_widget(nullptr),
      p_buttonBox(nullptr),
      p_statusLabel(nullptr),
      p_mainLayout(nullptr),
      p_progressBar(nullptr),
      p_progressLabel(nullptr),
      p_cancelButton(nullptr),
      p_progressTimer(nullptr),
      p_timeoutTimer(nullptr),
      d_mode(Mode::Settings),
      d_overlayType(overlay ? overlay->type() : OverlayBase::BCExperiment),
      d_overlay(overlay),
      d_dialogState(DialogState::Ready),
      d_operationProgress(0),
      d_isValid(true) // Settings mode starts valid
{
    setupUI();
    
    // Setup widget for settings mode
    p_widget->setupForSettings(overlay, plotNames, currentFt, overlayStorage);
    
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
    // Auto-preview in creation context - check if preview overlay exists
    return p_widget && p_widget->getPreviewOverlay() != nullptr;
}

void UnifiedOverlayDialog::accept()
{
    // Prevent multiple accept attempts
    if (d_dialogState == DialogState::Processing || d_dialogState == DialogState::Cancelling) {
        return;
    }
    
    if (!d_isValid) {
        QMessageBox::warning(this, "Validation Error", d_validationMessage);
        return;
    }
    
    // Save settings for all SettingsStorage-enabled widgets before proceeding
    if (p_widget) {
        p_widget->onAccept();
    }
    
    if (isCreationMode()) {
        // Analyze preview state and choose optimal creation path
        PreviewState state = analyzePreviewState();
        switch (state) {
        case PreviewState::CurrentPreview:
            // Fast path: use existing preview overlay
            finalizeFromPreview();
            break;
        case PreviewState::NoPreview:
        case PreviewState::StalePreview:
            // Standard path: create overlay in background
            createOverlayAsync();
            break;
        case PreviewState::ProcessingPreview:
            // Wait for preview to complete, then finalize
            QMessageBox::information(this, "Preview Processing", 
                "Please wait for preview processing to complete before creating the overlay.");
            return;
        }
    } else if (isSettingsMode()) {
        // Apply settings, potentially with background operations
        applySettingsAsync();
    }
}

void UnifiedOverlayDialog::reject()
{
    // Handle cancellation based on current state
    switch (d_dialogState) {
    case DialogState::Processing:
        // Cancel the background operation
        if (!d_currentOperationId.isEmpty()) {
            setDialogState(DialogState::Cancelling);
            auto& manager = OverlayProcessManager::instance();
            if (manager.cancelOperation(d_currentOperationId)) {
                // Operation cancelled successfully
                resetDialogState();
            } else {
                // Force reset if cancellation failed
                resetDialogState();
            }
        }
        return; // Don't close dialog immediately
        
    case DialogState::Cancelling:
        // Still cancelling - ignore reject
        return;
        
    case DialogState::Ready:
    case DialogState::Error:
    case DialogState::Complete:
    default:
        // Normal cancellation
        break;
    }
    
    // Clean up preview overlay explicitly before widget destruction to avoid race conditions
    if (isCreationMode() && p_widget) {
        p_widget->cleanupPreviewOverlay();
    }
    
    // In settings mode, restore the original overlay state before cancelling
    if (isSettingsMode() && p_widget && d_overlay) {
        p_widget->restoreOverlayState();
        
        // Emit signal to update plot display with restored values
        emit overlayDataChanged(d_overlay);
    }
    
    // Clean up any pending operations
    if (!d_currentOperationId.isEmpty()) {
        auto& manager = OverlayProcessManager::instance();
        manager.cancelOperation(d_currentOperationId);
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
    // Get the preview overlay from the widget
    auto previewOverlay = p_widget->getPreviewOverlay();
    if (previewOverlay) {
        // Emit signal for overlay manager to handle preview addition
        emit previewOverlayRequested(previewOverlay);
    }
    
    // Forward signal and update button states
    emit previewRequested();
    updateButtonState();
    updateWindowTitle();
}

void UnifiedOverlayDialog::onPreviewCancelled()
{
    // Get the preview overlay from the widget and signal for removal
    auto previewOverlay = p_widget->getPreviewOverlay();
    if (previewOverlay) {
        emit previewOverlayCancelled(previewOverlay);
    }
    
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
    
    // Create the unified overlay widget with appropriate context
    QString settingsKey = QString("UnifiedOverlayDialog_%1").arg(static_cast<int>(d_overlayType));
    auto context = isCreationMode() ? UnifiedOverlayWidget::Context::Creation : UnifiedOverlayWidget::Context::Settings;
    p_widget = new UnifiedOverlayWidget(settingsKey, context, this);
    p_mainLayout->addWidget(p_widget, 1); // Give it all available space
    
    // Create status label
    p_statusLabel = new QLabel(this);
    p_statusLabel->setStyleSheet("QLabel { color: gray; font-style: italic; }");
    p_statusLabel->setText("Validating settings...");
    p_mainLayout->addWidget(p_statusLabel);
    
    // Create progress components (initially hidden)
    p_progressBar = new QProgressBar(this);
    p_progressBar->setRange(0, 100);
    p_progressBar->setVisible(false);
    p_mainLayout->addWidget(p_progressBar);
    
    p_progressLabel = new QLabel(this);
    p_progressLabel->setVisible(false);
    p_mainLayout->addWidget(p_progressLabel);
    
    // Create timers
    p_progressTimer = new QTimer(this);
    p_progressTimer->setSingleShot(false);
    connect(p_progressTimer, &QTimer::timeout, this, &UnifiedOverlayDialog::updateProgressDisplay);
    
    p_timeoutTimer = new QTimer(this);
    p_timeoutTimer->setSingleShot(true);
    connect(p_timeoutTimer, &QTimer::timeout, this, &UnifiedOverlayDialog::onOperationTimeout);
    
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
    
    // Real-time overlay updates
    connect(p_widget, &UnifiedOverlayWidget::overlayDataChanged,
                this, &UnifiedOverlayDialog::onOverlayDataChanged);
}

void UnifiedOverlayDialog::updateButtonState()
{
    if (!p_buttonBox) {
        return;
    }
    
    QPushButton *okButton = p_buttonBox->button(QDialogButtonBox::Ok);
    QPushButton *cancelButton = p_buttonBox->button(QDialogButtonBox::Cancel);
    
    if (okButton) {
        switch (d_dialogState) {
        case DialogState::Ready:
            // Normal state - enable based on validation
            okButton->setEnabled(isSettingsMode() || d_isValid);
            okButton->setText(getOkButtonText());
            break;
            
        case DialogState::Processing:
            // Disable OK during processing
            okButton->setEnabled(false);
            okButton->setText("Processing...");
            break;
            
        case DialogState::Cancelling:
            // Keep disabled during cancellation
            okButton->setEnabled(false);
            okButton->setText("Cancelling...");
            break;
            
        case DialogState::Error:
            // Enable OK to retry or cancel to abort
            okButton->setEnabled(true);
            okButton->setText("Retry");
            break;
            
        case DialogState::Complete:
            // Enable OK to finalize
            okButton->setEnabled(true);
            okButton->setText("OK");
            break;
        }
    }
    
    if (cancelButton) {
        switch (d_dialogState) {
        case DialogState::Ready:
        case DialogState::Error:
        case DialogState::Complete:
            cancelButton->setEnabled(true);
            cancelButton->setText("Cancel");
            break;
            
        case DialogState::Processing:
            cancelButton->setEnabled(true);
            cancelButton->setText("Cancel Operation");
            break;
            
        case DialogState::Cancelling:
            cancelButton->setEnabled(false);
            cancelButton->setText("Cancelling...");
            break;
        }
    }
}

QString UnifiedOverlayDialog::getOkButtonText() const
{
    if (d_dialogState != DialogState::Ready) {
        return "OK";
    }
    
    if (isInPreviewMode()) {
        return isCreationMode() ? "Create from Preview" : "Apply from Preview";
    } else {
        return isCreationMode() ? "Create Overlay" : "Apply Changes";
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

OverlayTypeSpecificWidget* UnifiedOverlayDialog::getTypeSpecificWidget() const
{
    return p_widget ? p_widget->getTypeSpecificWidget() : nullptr;
}

bool UnifiedOverlayDialog::isCreationMode() const
{
    return d_mode == Mode::Creation;
}

bool UnifiedOverlayDialog::isSettingsMode() const
{
    return d_mode == Mode::Settings;
}

UnifiedOverlayDialog::PreviewState UnifiedOverlayDialog::analyzePreviewState() const
{
    if (!p_widget) {
        return PreviewState::NoPreview;
    }
    
    if (!isInPreviewMode()) {
        return PreviewState::NoPreview;
    }
    
    // In auto-preview architecture, previews are always current
    // (Phase 2 will implement intelligent sync checking)
    
    // Check if a background operation is currently updating the preview
    // (For now, we'll use a simplified check - in full implementation this would 
    // check the OverlayProcessManager for pending preview operations)
    if (d_dialogState == DialogState::Processing) {
        return PreviewState::ProcessingPreview;
    }
    
    return PreviewState::CurrentPreview;
}

void UnifiedOverlayDialog::createOverlayAsync()
{
    if (!p_widget) {
        return;
    }
    
    setDialogState(DialogState::Processing);
    
    // Use the operation declaration interface to determine processing strategy
    auto typeSpecificWidget = getTypeSpecificWidget();
    if (typeSpecificWidget && typeSpecificWidget->supportsBackgroundOperation(OperationCapability::Creation)) {
        // Check if background creation is beneficial for this widget type
        d_createdOverlay = p_widget->createOverlay();
        if (!d_createdOverlay) {
            setDialogState(DialogState::Error);
            d_operationError = "Failed to create overlay from current settings";
            updateButtonState();
            return;
        }
        
        // Create background operation using the widget's factory method
        auto operation = typeSpecificWidget->createOperation(OperationCapability::Creation, d_createdOverlay);
        if (operation) {
            // Use background processing
            auto& manager = OverlayProcessManager::instance();
            d_currentOperationId = manager.queueOperation(operation, OverlayProcessManager::Priority::High);
            
            // Connect to manager signals for this specific operation
            connect(&manager, &OverlayProcessManager::operationStarted,
                    this, &UnifiedOverlayDialog::onOperationStarted);
            connect(&manager, &OverlayProcessManager::operationProgress,
                    this, &UnifiedOverlayDialog::onOperationProgress);
            connect(&manager, &OverlayProcessManager::operationCompleted,
                    this, &UnifiedOverlayDialog::onOperationCompleted);
            connect(&manager, &OverlayProcessManager::operationFailed,
                    this, &UnifiedOverlayDialog::onOperationFailed);
            connect(&manager, &OverlayProcessManager::operationCancelled,
                    this, &UnifiedOverlayDialog::onOperationCancelled);
        } else {
            // Widget returned nullptr - use synchronous completion
            setDialogState(DialogState::Complete);
            QDialog::accept();
        }
    } else {
        // Use synchronous creation for widgets that don't support background processing
        d_createdOverlay = p_widget->createOverlay();
        if (d_createdOverlay) {
            setDialogState(DialogState::Complete);
            QDialog::accept();
        } else {
            setDialogState(DialogState::Error);
            d_operationError = "Failed to create overlay from current settings";
            updateButtonState();
        }
    }
}

void UnifiedOverlayDialog::finalizeFromPreview()
{
    if (!p_widget || !isInPreviewMode()) {
        createOverlayAsync();
        return;
    }
    
    // Fast path: get preview overlay and clear preview flag  
    auto previewOverlay = p_widget->getPreviewOverlay();
    if (previewOverlay) {
        d_createdOverlay = previewOverlay;
        d_createdOverlay->setPreview(false);
        
        // Clear the preview reference so the widget destructor doesn't disable the overlay
        p_widget->clearPreviewOverlay();
        
        QDialog::accept();
    } else {
        createOverlayAsync();
    }
}

void UnifiedOverlayDialog::applySettingsAsync()
{
    if (!p_widget) {
        return;
    }
    
    setDialogState(DialogState::Processing);
    
    // Apply current settings to overlay
    p_widget->applyToOverlay();
    
    // Auto-preview cleanup will be handled automatically
    
    setDialogState(DialogState::Complete);
    QDialog::accept();
}

void UnifiedOverlayDialog::setDialogState(DialogState state)
{
    if (d_dialogState == state) {
        return;
    }
    
    d_dialogState = state;
    updateButtonState();
    updateWindowTitle();
    
    // Handle state-specific logic
    switch (state) {
    case DialogState::Ready:
        if (p_progressBar) p_progressBar->setVisible(false);
        if (p_progressLabel) p_progressLabel->setVisible(false);
        if (p_cancelButton) p_cancelButton->setVisible(false);
        if (p_progressTimer) p_progressTimer->stop();
        if (p_timeoutTimer) p_timeoutTimer->stop();
        break;
        
    case DialogState::Processing:
        if (p_progressBar) {
            p_progressBar->setVisible(true);
            p_progressBar->setValue(0);
        }
        if (p_progressLabel) {
            p_progressLabel->setVisible(true);
            p_progressLabel->setText("Processing...");
        }
        if (p_cancelButton) p_cancelButton->setVisible(true);
        if (p_progressTimer) p_progressTimer->start(PROGRESS_UPDATE_INTERVAL_MS);
        if (p_timeoutTimer) p_timeoutTimer->start(OPERATION_TIMEOUT_MS);
        break;
        
    case DialogState::Cancelling:
        if (p_progressLabel) p_progressLabel->setText("Cancelling...");
        if (p_cancelButton) p_cancelButton->setEnabled(false);
        break;
        
    case DialogState::Error:
        if (p_progressBar) p_progressBar->setVisible(false);
        if (p_progressLabel) {
            p_progressLabel->setVisible(true);
            p_progressLabel->setText(QString("Error: %1").arg(d_operationError));
            p_progressLabel->setStyleSheet("QLabel { color: red; }");
        }
        if (p_cancelButton) p_cancelButton->setVisible(false);
        if (p_progressTimer) p_progressTimer->stop();
        if (p_timeoutTimer) p_timeoutTimer->stop();
        break;
        
    case DialogState::Complete:
        if (p_progressBar) {
            p_progressBar->setVisible(true);
            p_progressBar->setValue(100);
        }
        if (p_progressLabel) {
            p_progressLabel->setVisible(true);
            p_progressLabel->setText("Operation completed successfully");
            p_progressLabel->setStyleSheet("QLabel { color: green; }");
        }
        if (p_cancelButton) p_cancelButton->setVisible(false);
        if (p_progressTimer) p_progressTimer->stop();
        if (p_timeoutTimer) p_timeoutTimer->stop();
        break;
    }
}

void UnifiedOverlayDialog::resetDialogState()
{
    d_currentOperationId.clear();
    d_operationError.clear();
    d_operationProgress = 0;
    d_operationMessage.clear();
    setDialogState(DialogState::Ready);
}

// Background operation handlers (stubs for now)
void UnifiedOverlayDialog::onOperationStarted(const QString &operationId)
{
    d_currentOperationId = operationId;
    setDialogState(DialogState::Processing);
}

void UnifiedOverlayDialog::onOperationProgress(const QString &operationId, int percentage, const QString &message)
{
    if (operationId != d_currentOperationId) {
        return;
    }
    
    d_operationProgress = percentage;
    d_operationMessage = message;
    updateProgressDisplay();
}

void UnifiedOverlayDialog::onOperationCompleted(const QString &operationId, std::shared_ptr<OverlayBase> result)
{
    if (operationId != d_currentOperationId) {
        return;
    }
    
    if (isCreationMode()) {
        d_createdOverlay = result;
    }
    
    setDialogState(DialogState::Complete);
    
    // Auto-close after brief delay
    QTimer::singleShot(1000, this, &UnifiedOverlayDialog::accept);
}

void UnifiedOverlayDialog::onOperationFailed(const QString &operationId, const QString &error)
{
    if (operationId != d_currentOperationId) {
        return;
    }
    
    d_operationError = error;
    setDialogState(DialogState::Error);
}

void UnifiedOverlayDialog::onOperationCancelled(const QString &operationId)
{
    if (operationId != d_currentOperationId) {
        return;
    }
    
    resetDialogState();
}

void UnifiedOverlayDialog::updateProgressDisplay()
{
    if (p_progressBar) {
        p_progressBar->setValue(d_operationProgress);
    }
    
    if (p_progressLabel && !d_operationMessage.isEmpty()) {
        p_progressLabel->setText(d_operationMessage);
    }
}

void UnifiedOverlayDialog::onOperationTimeout()
{
    if (d_dialogState == DialogState::Processing) {
        d_operationError = "Operation timed out after 30 seconds";
        setDialogState(DialogState::Error);
        
        // Attempt to cancel the operation
        if (!d_currentOperationId.isEmpty()) {
            auto& manager = OverlayProcessManager::instance();
            manager.cancelOperation(d_currentOperationId);
        }
    }
}
