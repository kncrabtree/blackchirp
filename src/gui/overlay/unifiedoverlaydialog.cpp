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
                                         std::shared_ptr<OverlayStorage> overlayStorage,
                                         QWidget *parent)
    : QDialog(parent)
{
    initializeCommon(type, plotNames, currentFt, nullptr, overlayStorage);
    d_mode = Mode::Creation;
    d_isValid = false; // Creation mode starts invalid
}

// Settings mode constructor
UnifiedOverlayDialog::UnifiedOverlayDialog(std::shared_ptr<OverlayBase> overlay,
                                         const QStringList &plotNames,
                                         const Ft &currentFt,
                                         std::shared_ptr<OverlayStorage> overlayStorage,
                                         QWidget *parent)
    : QDialog(parent)
{
    OverlayBase::OverlayType type = overlay ? overlay->type() : OverlayBase::BCExperiment;
    initializeCommon(type, plotNames, currentFt, overlay, overlayStorage);
    d_mode = Mode::Settings;
    d_overlay = overlay;
    d_isValid = true; // Settings mode starts valid
}

UnifiedOverlayDialog::~UnifiedOverlayDialog() = default;

void UnifiedOverlayDialog::initializeCommon(OverlayBase::OverlayType type, 
                                           const QStringList &plotNames,
                                           const Ft &currentFt,
                                           std::shared_ptr<OverlayBase> overlay,
                                           std::shared_ptr<OverlayStorage> overlayStorage)
{
    // Initialize UI component pointers
    p_widget = nullptr;
    p_buttonBox = nullptr;
    p_statusLabel = nullptr;
    p_mainLayout = nullptr;
    p_progressBar = nullptr;
    p_progressLabel = nullptr;
    p_cancelButton = nullptr;
    p_progressTimer = nullptr;
    p_timeoutTimer = nullptr;
    
    // Initialize state variables
    d_overlayType = type;
    d_overlayStorage = overlayStorage;
    d_dialogState = DialogState::Ready;
    d_operationProgress = 0;
    d_queueSize = 0;
    d_isProcessing = false;
    
    // Create widget with all necessary data (context auto-detected from overlay parameter)
    QString settingsKey = QString("UnifiedOverlayDialog_%1").arg(static_cast<int>(d_overlayType));
    p_widget = new UnifiedOverlayWidget(settingsKey, d_overlayType, plotNames, currentFt, 
                                        overlay, overlayStorage, this);
    
    setupUI();
    setupConnections();
    updateWindowTitle();
    updateButtonState();
}

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
    
    // Check for unsaved changes and allow type-specific widgets to handle validation
    if (p_widget && !p_widget->validateAcceptance()) {
        return; // Type-specific widget indicated not to proceed
    }
    
    // Block signals to prevent race conditions with background plot operations
    if (p_widget) {
        p_widget->blockSignals(true);
    }
    
    // Save settings for all SettingsStorage-enabled widgets before proceeding
    if (p_widget) {
        p_widget->onAccept();
    }
    
    if (isCreationMode()) {
        // Check if we have a valid preview overlay ready to use
        auto previewOverlay = p_widget->getPreviewOverlay();
        if (previewOverlay) {
            // Fast path: preview overlay exists and should be fully processed
            // Convert to final overlay by clearing preview flag
            d_createdOverlay = previewOverlay;
            d_createdOverlay->setPreview(false);
            
            // Clear the preview reference so widget destructor doesn't disable the overlay
            p_widget->clearPreviewOverlay();
            
            QDialog::accept();
        } else {
            // Fallback: create overlay directly (should be rare in auto-preview mode)
            d_createdOverlay = p_widget->createOverlay();
            if (d_createdOverlay) {
                QDialog::accept();
            } else {
                // Unblock signals before showing error (dialog stays open)
                if (p_widget) {
                    p_widget->blockSignals(false);
                }
                QMessageBox::warning(this, "Creation Error", 
                    "Failed to create overlay from current settings");
            }
        }
    } else if (!isCreationMode()) {
        // Check if overlay label has changed and handle renaming if needed
        if (d_overlayStorage && p_widget) {
            QString originalLabel = p_widget->getOriginalLabel();
            QString currentLabel = d_overlay ? d_overlay->getLabel() : QString();
            
            if (!originalLabel.isEmpty() && !currentLabel.isEmpty() && originalLabel != currentLabel) {
                if (!d_overlayStorage->renameOverlay(originalLabel, currentLabel)) {
                    // Unblock signals before showing error and returning
                    if (p_widget) {
                        p_widget->blockSignals(false);
                    }
                    QMessageBox::warning(this, "Rename Failed", 
                                        QString("Failed to rename overlay from '%1' to '%2'. "
                                               "Please check that the new name is valid and not already in use.")
                                               .arg(originalLabel, currentLabel));
                    return; // Don't close dialog on rename failure
                }
            }
        }
        
        // Apply settings directly
        setDialogState(DialogState::Processing);
        
        // Apply current settings to overlay
        p_widget->applyToOverlay();
        
        QDialog::accept();
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
    default:
        // Normal cancellation
        break;
    }
    
    // Clean up preview overlay explicitly before widget destruction to avoid race conditions
    if (isCreationMode() && p_widget) {
        p_widget->cleanupPreviewOverlay();
    }
    
    // In settings mode, restore the original overlay state before cancelling
    if (!isCreationMode() && p_widget && d_overlay) {
        p_widget->restoreOverlayState();
        
        // Emit signal to update plot display with restored values
        emit overlayDataChanged(d_overlay);
    }
    
    // Clean up any pending operations
    if (!d_currentOperationId.isEmpty()) {
        auto& manager = OverlayProcessManager::instance();
        manager.cancelOperation(d_currentOperationId);
    }

    // Ensure signals are unblocked before closing
    if (p_widget) {
        p_widget->blockSignals(false);
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
    handlePreviewChange(true);
}

void UnifiedOverlayDialog::onPreviewCancelled()
{
    handlePreviewChange(false);
}

void UnifiedOverlayDialog::handlePreviewChange(bool isRequested)
{
    // Get the preview overlay from the widget
    auto previewOverlay = p_widget->getPreviewOverlay();
    if (previewOverlay) {
        // Emit appropriate signal for overlay manager
        if (isRequested) {
            emit previewOverlayRequested(previewOverlay);
        } else {
            emit previewOverlayCancelled(previewOverlay);
        }
    }
    
    // Forward appropriate signal and update UI
    if (isRequested) {
        emit previewRequested();
    } else {
        emit previewCancelled();
    }
    
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
    resize(800, 400);
    
    // Create main layout
    p_mainLayout = new QVBoxLayout(this);
    p_mainLayout->setContentsMargins(12, 12, 12, 12);
    p_mainLayout->setSpacing(12);
    
    // Widget is already created in constructor with all necessary data
    p_mainLayout->addWidget(p_widget, 1); // Give it all available space
    
    // Create status label
    p_statusLabel = new QLabel(this);
    p_statusLabel->setStyleSheet("QLabel { color: gray; font-style: italic; }");
    p_statusLabel->setText("Validating settings...");
    p_mainLayout->addWidget(p_statusLabel);
    
    // Create progress components (initially hidden)
    p_progressBar = new QProgressBar(this);
    p_progressBar->setRange(0, 100);
    p_progressBar->setValue(100);
    p_mainLayout->addWidget(p_progressBar);
    
    p_progressLabel = new QLabel(this);
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
    
    // Connect to OverlayProcessManager for background operation progress
    auto& manager = OverlayProcessManager::instance();
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
    connect(&manager, &OverlayProcessManager::queueSizeChanged,
            this, &UnifiedOverlayDialog::onQueueSizeChanged);
    connect(&manager, &OverlayProcessManager::processingStateChanged,
            this, &UnifiedOverlayDialog::onProcessingStateChanged);
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
            // Normal state - enable based on validation and no pending operations
            // Note: We track processing state through signals to avoid mutex deadlock
            okButton->setEnabled(d_isValid && d_queueSize == 0 && !d_isProcessing);
            okButton->setText(isCreationMode() ? "Create Overlay" : "Apply Changes");
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
        }
    }
    
    if (cancelButton) {
        switch (d_dialogState) {
        case DialogState::Ready:
        case DialogState::Error:
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
        if (p_cancelButton) p_cancelButton->setVisible(false);
        if (p_progressTimer) p_progressTimer->stop();
        if (p_timeoutTimer) p_timeoutTimer->stop();
        break;
        
    case DialogState::Processing:
        if (p_progressBar) {
            p_progressBar->setValue(0);
        }
        if (p_progressLabel) {
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
        if (p_progressLabel) {
            p_progressLabel->setText(QString("Error: %1").arg(d_operationError));
            p_progressLabel->setStyleSheet("QLabel { color: red; }");
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
    Q_UNUSED(result); // Background operations update widgets directly, not via dialog
    
    if (operationId != d_currentOperationId) {
        return;
    }
    
    // Clear the current operation
    d_currentOperationId.clear();
    
    // Update progress display to show completion
    if (p_progressBar) {
        p_progressBar->setValue(100);
    }
    if (p_progressLabel) {
        p_progressLabel->setText("Operation completed successfully");
        p_progressLabel->setStyleSheet("QLabel { color: green; }");
    }
    
    // Return to ready state - user can now interact with dialog normally
    setDialogState(DialogState::Ready);
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

void UnifiedOverlayDialog::onQueueSizeChanged(int size)
{
    // Update cached queue size to avoid mutex deadlock
    d_queueSize = size;
    
    // Update status label to show queue information
    if (p_statusLabel) {
        if (size > 0) {
            p_statusLabel->setText(QString("Background operations: %1 pending").arg(size));
            p_statusLabel->setStyleSheet("QLabel { color: blue; }");
        } else if (d_isValid) {
            p_statusLabel->setText("Settings are valid");
            p_statusLabel->setStyleSheet("QLabel { color: green; }");
        }
    }
    
    // Update button state - disable OK button if operations are pending
    updateButtonState();
}

void UnifiedOverlayDialog::onProcessingStateChanged(bool isProcessing)
{
    // Update cached processing state to avoid mutex deadlock
    d_isProcessing = isProcessing;
    
    // Update dialog state based on overall processing status
    if (isProcessing && d_dialogState == DialogState::Ready) {
        // Show that background processing is active
        if (p_statusLabel) {
            p_statusLabel->setText("Background processing active...");
            p_statusLabel->setStyleSheet("QLabel { color: blue; }");
        }
    } else if (!isProcessing && d_dialogState == DialogState::Ready) {
        // Processing finished - restore normal validation status
        if (p_statusLabel && d_isValid) {
            p_statusLabel->setText("Settings are valid");
            p_statusLabel->setStyleSheet("QLabel { color: green; }");
        }
    }
    
    // Update button state
    updateButtonState();
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
