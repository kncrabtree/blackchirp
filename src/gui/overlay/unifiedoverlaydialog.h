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
#include <set>

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
    // State management enum
    enum class DialogState {
        Ready,        // Normal state, can accept user input
        Processing,   // Background operation in progress  
        Cancelling,   // Gracefully aborting operations
        Error         // Operation failed, showing error state
    };

    // Constructors and destructor
    explicit UnifiedOverlayDialog(OverlayBase::OverlayType type,
                                 const QStringList &plotNames,
                                 const Ft &currentFt,
                                 std::shared_ptr<OverlayStorage> overlayStorage,
                                 QWidget *parent = nullptr);
    
    explicit UnifiedOverlayDialog(std::shared_ptr<OverlayBase> overlay,
                                 const QStringList &plotNames,
                                 const Ft &currentFt,
                                 std::shared_ptr<OverlayStorage> overlayStorage,
                                 QWidget *parent = nullptr);
    
    ~UnifiedOverlayDialog();
    
    // Core interface methods
    std::shared_ptr<OverlayBase> getOverlay() const;
    bool isInPreviewMode() const;
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
    // Widget signal handlers
    void onValidationStatusChanged(bool isValid, const QString &message);
    void onPreviewRequested();
    void onPreviewCancelled(std::shared_ptr<OverlayBase> overlay);
    void onOverlayDataChanged(std::shared_ptr<OverlayBase> overlay);
    
    // Background operation handlers
    void onOperationQueued(const QString &operationId);
    void onOperationStarted(const QString &operationId);
    void onOperationProgress(const QString &operationId, int percentage, const QString &message);
    void onOperationCompleted(const QString &operationId, std::shared_ptr<OverlayBase> result);
    void onOperationFailed(const QString &operationId, const QString &error);
    void onOperationCancelled(const QString &operationId);
    void onQueueSizeChanged(int size);
    void onProcessingStateChanged(bool isProcessing);
    
    // Timer handlers
    void updateProgressDisplay();

private:
    // Constants
    static const int PROGRESS_UPDATE_INTERVAL_MS = 100;
    
    // Context mode enum
    enum class Mode {
        Creation,
        Settings
    };

    // UI setup methods
    void setupUI();
    void setupConnections();
    void updateButtonState();
    void updateWindowTitle();
    
    // State management methods
    void setDialogState(DialogState state);
    void resetDialogState();
    
    // Helper methods
    QString getContextName() const;
    QString getTypeName() const;
    bool isCreationMode() const;
    
    // Common initialization
    void initializeCommon(OverlayBase::OverlayType type, 
                         const QStringList &plotNames,
                         const Ft &currentFt,
                         std::shared_ptr<OverlayBase> overlay,
                         std::shared_ptr<OverlayStorage> overlayStorage);
    
    // Core UI components
    UnifiedOverlayWidget *p_widget;
    QDialogButtonBox *p_buttonBox;
    QLabel *p_statusLabel;
    QVBoxLayout *p_mainLayout;
    
    // Progress UI components
    QProgressBar *p_progressBar;
    QLabel *p_progressLabel;
    QPushButton *p_cancelButton;
    
    // Timers
    QTimer *p_progressTimer;
    
    // Context and configuration
    Mode d_mode;
    OverlayBase::OverlayType d_overlayType;
    std::shared_ptr<OverlayBase> d_overlay; // Settings mode only
    std::shared_ptr<OverlayBase> d_createdOverlay; // Creation mode result
    std::shared_ptr<OverlayStorage> d_overlayStorage;
    
    // Dialog state tracking
    DialogState d_dialogState;
    bool d_busyCursorActive = false; // Balances setOverrideCursor/restore
    QString d_currentOperationId;
    // Every operation id this dialog's type-specific widget has queued
    // (via OverlayTypeSpecificWidget::operationQueued), pruned as each
    // one reaches a terminal state. OverlayProcessManager is a
    // process-wide singleton whose operationStarted() fires for every
    // operation from every open overlay dialog; this set is what lets
    // onOperationStarted() tell "mine" from "someone else's".
    std::set<QString, std::less<>> d_ownedOperationIds;
    QString d_operationError;
    int d_operationProgress;
    QString d_operationMessage;
    
    // Operation state tracking (to avoid mutex deadlock)
    int d_queueSize;
    bool d_isProcessing;
    
    // Validation state
    bool d_isValid;
    QString d_validationMessage;
};

#endif // UNIFIEDOVERLAYDIALOG_H
