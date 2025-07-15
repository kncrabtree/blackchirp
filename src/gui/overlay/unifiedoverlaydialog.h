#ifndef UNIFIEDOVERLAYDIALOG_H
#define UNIFIEDOVERLAYDIALOG_H

#include <QDialog>
#include <QDialogButtonBox>
#include <QVBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QTimer>
#include <memory>

#include <data/experiment/overlaybase.h>
#include <data/storage/overlaystorage.h>
#include <data/processing/overlayprocessmanager.h>
#include <data/analysis/ft.h>

// Forward declarations
class UnifiedOverlayWidget;
class OverlayTypeSpecificWidget;

/**
 * @brief Single unified dialog for both overlay creation and configuration
 * 
 * This dialog replaces both OverlayConfigDialog and OverlaySettingsDialog,
 * providing a consistent interface that adapts based on whether it's being
 * used for creation or settings modification. It uses the UnifiedOverlayWidget
 * internally and provides preview mode functionality for superior user experience.
 */
class UnifiedOverlayDialog : public QDialog
{
    Q_OBJECT

public:
    // Creation mode constructor
    explicit UnifiedOverlayDialog(OverlayBase::OverlayType type,
                                 const QStringList &plotNames,
                                 const Ft &currentFt,
                                 const QVector<std::shared_ptr<OverlayBase>> &existingOverlays = {},
                                 QWidget *parent = nullptr);
    
    // Settings mode constructor
    explicit UnifiedOverlayDialog(std::shared_ptr<OverlayBase> overlay,
                                 const QStringList &plotNames,
                                 const Ft &currentFt,
                                 std::shared_ptr<OverlayStorage> overlayStorage,
                                 QWidget *parent = nullptr);
    
    ~UnifiedOverlayDialog();
    
    // Get the created overlay (creation mode only)
    std::shared_ptr<OverlayBase> getOverlay() const;
    
    // Check if preview mode is active
    bool isInPreviewMode() const;
    
    // State management
    enum class DialogState {
        Ready,        // Normal state, can accept user input
        Processing,   // Background operation in progress  
        Cancelling,   // Gracefully aborting operations
        Error,        // Operation failed, showing error state
        Complete      // Operation completed successfully
    };
    
    DialogState getDialogState() const { return d_dialogState; }

public slots:
    void accept() override;
    void reject() override;

signals:
    // Preview mode signals (forwarded from widget)
    void previewRequested();
    void previewCancelled();
    void overlayDataChanged(std::shared_ptr<OverlayBase> overlay);
    
    // Preview overlay management signals
    void previewOverlayRequested(std::shared_ptr<OverlayBase> overlay);
    void previewOverlayCancelled(std::shared_ptr<OverlayBase> overlay);

private slots:
    void onValidationStatusChanged(bool isValid, const QString &message);
    void onPreviewRequested();
    void onPreviewCancelled();
    void onOverlayDataChanged(std::shared_ptr<OverlayBase> overlay);
    
    // Background operation handling
    void onOperationStarted(const QString &operationId);
    void onOperationProgress(const QString &operationId, int percentage, const QString &message);
    void onOperationCompleted(const QString &operationId, std::shared_ptr<OverlayBase> result);
    void onOperationFailed(const QString &operationId, const QString &error);
    void onOperationCancelled(const QString &operationId);
    
    // UI update timers
    void updateProgressDisplay();
    void onOperationTimeout();

private:
    // UI setup
    void setupUI();
    void setupConnections();
    void updateButtonState();
    void updateWindowTitle();
    
    // Helper methods
    QString getContextName() const;
    QString getTypeName() const;
    QString getOkButtonText() const;
    bool isCreationMode() const;
    bool isSettingsMode() const;
    
    // Type-specific widget access for operation declaration interface
    OverlayTypeSpecificWidget* getTypeSpecificWidget() const;
    
    // Preview state analysis for smart creation workflow
    enum class PreviewState {
        NoPreview,        // Never created preview
        CurrentPreview,   // Preview exists and matches settings
        StalePreview,     // Preview exists but out of sync
        ProcessingPreview // Preview being updated in background
    };
    PreviewState analyzePreviewState() const;
    
    // Async workflow methods
    void createOverlayAsync();
    void finalizeFromPreview();
    void applySettingsAsync();
    
    // State management
    void setDialogState(DialogState state);
    void resetDialogState();
    
    // Core components
    UnifiedOverlayWidget *p_widget;
    QDialogButtonBox *p_buttonBox;
    QLabel *p_statusLabel;
    QVBoxLayout *p_mainLayout;
    
    // Progress indication
    QProgressBar *p_progressBar;
    QLabel *p_progressLabel;
    QPushButton *p_cancelButton;
    QTimer *p_progressTimer;
    QTimer *p_timeoutTimer;
    
    // Context and state
    enum class Mode {
        Creation,
        Settings
    } d_mode;
    
    OverlayBase::OverlayType d_overlayType;
    std::shared_ptr<OverlayBase> d_overlay; // Settings mode only
    std::shared_ptr<OverlayBase> d_createdOverlay; // Creation mode result
    
    // Dialog state management
    DialogState d_dialogState;
    QString d_currentOperationId;
    QString d_operationError;
    int d_operationProgress;
    QString d_operationMessage;
    
    // Validation state
    bool d_isValid;
    QString d_validationMessage;
    
    // Configuration
    static const int OPERATION_TIMEOUT_MS = 30000; // 30 seconds
    static const int PROGRESS_UPDATE_INTERVAL_MS = 100;
};

#endif // UNIFIEDOVERLAYDIALOG_H
