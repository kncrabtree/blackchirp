#ifndef BCEXPOVERLAYWIDGET_H
#define BCEXPOVERLAYWIDGET_H

#include <QSpinBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QToolButton>
#include <QFormLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <memory>

#include <data/analysis/ft.h>
#include <data/experiment/overlaytypes.h>
#include "overlaytypespecificwidget.h"

class ExperimentViewWidget;
class FtmwViewWidget;

/**
 * @brief Type-specific widget for BCExperiment overlays
 * 
 * This widget handles all BCExperiment-specific functionality within the
 * UnifiedOverlayWidget architecture. It provides experiment selection,
 * FT configuration, and source file management specifically for BlackChirp
 * experiment data.
 */
class BCExpOverlayWidget : public OverlayTypeSpecificWidget
{
    Q_OBJECT

public:
    explicit BCExpOverlayWidget(const Ft &currentFt, QWidget *parent = nullptr);
    ~BCExpOverlayWidget();

    // OverlayTypeSpecificWidget interface
    void setupForCreation() override;
    void setupForSettings(std::shared_ptr<OverlayBase> overlay) override;
    
    std::shared_ptr<OverlayBase> createOverlay() override;
    void applyToOverlay(std::shared_ptr<OverlayBase> overlay) const override;
    
    bool validateSettingsImpl() override;
    bool isDataValid() const override;
    
    // Source file management
    bool hasValidSourceFile() const override;
    QString getSourceFilePath() const override;
    void setSourceFilePath(const QString &path) override;
    bool validateSourceFileImpl() override;
    
    void resetToDefaults() override;
    
    // Settings state capture for preview sync tracking
    QHash<QString, QVariant> getSettingsHash() const override;
    
    // Operation declaration interface
    constexpr QVector<OperationCapability> getSupportedOperations() const override
    {
        // BCExperiment overlays support creation and validation
        return {
            OperationCapability(
                OperationCapability::Creation, 
                false,  // Not expensive - BCExp creation is typically fast
                100,    // ~100ms estimate 
                "Create BCExperiment overlay",
                OverlayProcessManager::Priority::Normal
            ),
            OperationCapability(
                OperationCapability::Validation,
                false,  // File validation is quick
                50,     // ~50ms estimate
                "Validate experiment files",
                OverlayProcessManager::Priority::High
            )
        };
    }
    
    constexpr bool supportsBackgroundOperation(OperationCapability::Type type) const override
    {
        // BCExperiment operations are generally fast, no background processing needed
        Q_UNUSED(type);
        return false;
    }
    
    std::shared_ptr<OverlayOperation> createOperation(OperationCapability::Type type,
                                                     std::shared_ptr<OverlayBase> overlay = nullptr) const override;
    
    // Three-tier UI component access
    // Three-tier layout is now handled internally
    
    // Type-specific settings visibility
    bool hasTypeSpecificSettings() const override { return false; } // BC experiments have no overlay-specific settings currently

private slots:
    void onExperimentNumberChanged(int number);
    void onUsePathToggled(bool enabled);
    void onBrowseButtonClicked();
    void onPathChanged();
    void onConfigureFtClicked();
    void validateExperiment();
    void updateAutomaticLabel();

protected:
    // Three-tier UI creation interface
    void createSourceFileConfigUI(QGroupBox *parent) override;
    void createSourceFileSettingsUI(QGroupBox *parent) override;
    void createTypeSpecificSettingsUI(QGroupBox *parent) override;
    
    // OverlayTypeSpecificWidget interface
    void setupConnections() override;
    void loadSettings() override;
    void saveSettings() override;
    
    // Context-aware UI behavior
    void configureForCreationContext() override;
    void configureForSettingsContext() override;

private:
    // Source File Configuration tier (File selection)
    QSpinBox *p_experimentNumberSpinBox;
    QCheckBox *p_usePathCheckBox;
    QLineEdit *p_pathLineEdit;
    QToolButton *p_browseButton;
    QLabel *p_experimentStatusLabel;
    
    // Source File Settings tier (FT Configuration - source-dependent)
    QPushButton *p_configureFtButton;
    QLabel *p_ftStatusLabel;
    // Future: Add BCExp-specific settings like processing options
    
    // State management
    bool d_experimentValid;
    bool d_hasFtData;
    Ft d_configuredFt;
    
    // Helper methods
    void setupExperimentSelectionUI();
    void setupFtConfigurationUI();
    void setupBCExpSettingsUI();
    void resetFtConfiguration();
    QString getExperimentPath() const;
    bool validateExperimentPath(const QString &path, QString &errorMessage);
    void updateExperimentStatus();
    void updateFtStatus();
};

#endif // BCEXPOVERLAYWIDGET_H