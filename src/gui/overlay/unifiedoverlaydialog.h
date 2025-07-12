#ifndef UNIFIEDOVERLAYDIALOG_H
#define UNIFIEDOVERLAYDIALOG_H

#include <QDialog>
#include <QDialogButtonBox>
#include <QVBoxLayout>
#include <QLabel>
#include <memory>

#include <data/experiment/overlaybase.h>
#include <data/storage/overlaystorage.h>

// Forward declarations
class UnifiedOverlayWidget;

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
                                 double xRangeMin, double xRangeMax,
                                 const QVector<std::shared_ptr<OverlayBase>> &existingOverlays = {},
                                 QWidget *parent = nullptr);
    
    // Settings mode constructor
    explicit UnifiedOverlayDialog(std::shared_ptr<OverlayBase> overlay,
                                 const QStringList &plotNames,
                                 double xRangeMin, double xRangeMax,
                                 std::shared_ptr<OverlayStorage> overlayStorage,
                                 QWidget *parent = nullptr);
    
    ~UnifiedOverlayDialog();
    
    // Get the created overlay (creation mode only)
    std::shared_ptr<OverlayBase> getOverlay() const;
    
    // Check if preview mode is active
    bool isInPreviewMode() const;

public slots:
    void accept() override;
    void reject() override;

signals:
    // Preview mode signals (forwarded from widget)
    void previewRequested();
    void previewCancelled();
    void overlayDataChanged(std::shared_ptr<OverlayBase> overlay);

private slots:
    void onValidationStatusChanged(bool isValid, const QString &message);
    void onPreviewRequested();
    void onPreviewCancelled();
    void onOverlayDataChanged(std::shared_ptr<OverlayBase> overlay);

private:
    // UI setup
    void setupUI();
    void setupConnections();
    void updateButtonState();
    void updateWindowTitle();
    
    // Helper methods
    QString getContextName() const;
    QString getTypeName() const;
    bool isCreationMode() const;
    bool isSettingsMode() const;
    
    // Core components
    UnifiedOverlayWidget *p_widget;
    QDialogButtonBox *p_buttonBox;
    QLabel *p_statusLabel;
    QVBoxLayout *p_mainLayout;
    
    // Context and state
    enum class Mode {
        Creation,
        Settings
    } d_mode;
    
    OverlayBase::OverlayType d_overlayType;
    std::shared_ptr<OverlayBase> d_overlay; // Settings mode only
    std::shared_ptr<OverlayBase> d_createdOverlay; // Creation mode result
    
    // Validation state
    bool d_isValid;
    QString d_validationMessage;
};

#endif // UNIFIEDOVERLAYDIALOG_H