#ifndef CATALOGOVERLAYWIDGET_H
#define CATALOGOVERLAYWIDGET_H

#include <QLineEdit>
#include <QToolButton>
#include <QLabel>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QFormLayout>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <memory>

#include <data/experiment/overlaytypes.h>
#include <data/experiment/catalogdata.h>
#include <data/experiment/catalogparserregistry.h>
#include <data/storage/settingsstorage.h>
#include <data/processing/overlayprocessmanager.h>
#include "overlaytypespecificwidget.h"

class FtmwViewWidget;

namespace BC::Key::CatalogWidget {
static const QString key{"CatalogOverlayWidget"};
static const QString lastFilePath{"lastFilePath"};
static const QString convolutionEnabled{"convolutionEnabled"};
static const QString lineshapeType{"lineshapeType"};
static const QString linewidthKHz{"linewidthKHz"};
static const QString minFreqMHz{"minFreqMHz"};
static const QString maxFreqMHz{"maxFreqMHz"};
static const QString pointSpacingMHz{"pointSpacingMHz"};
static const QString saveRangeOnly{"saveRangeOnly"};

// Metasettings for spinbox configuration
static const QString linewidthMin{"linewidthMin"};
static const QString linewidthMax{"linewidthMax"};
static const QString linewidthDecimals{"linewidthDecimals"};
static const QString linewidthStep{"linewidthStep"};
static const QString freqMin{"freqMin"};
static const QString freqMax{"freqMax"};
static const QString freqDecimals{"freqDecimals"};
static const QString freqStep{"freqStep"};
static const QString pointSpacingMin{"pointSpacingMin"};
static const QString pointSpacingMax{"pointSpacingMax"};
static const QString pointSpacingDecimals{"pointSpacingDecimals"};
static const QString pointSpacingStep{"pointSpacingStep"};
}

/**
 * @brief Type-specific widget for Catalog overlays
 * 
 * This widget handles all Catalog-specific functionality within the
 * UnifiedOverlayWidget architecture. It provides catalog file selection,
 * convolution settings, and source file management specifically for
 * spectroscopic catalog data (SPCAT, XIAM, etc.).
 */
class CatalogOverlayWidget : public OverlayTypeSpecificWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit CatalogOverlayWidget(QWidget *parent = nullptr);
    ~CatalogOverlayWidget();

    // OverlayTypeSpecificWidget interface
    void setupForCreation() override;
    void setupForSettings(std::shared_ptr<OverlayBase> overlay) override;
    
    std::shared_ptr<OverlayBase> createOverlay() const override;
    void applyToOverlay(std::shared_ptr<OverlayBase> overlay) const override;
    
    bool validateSettings(QString &errorMessage) const override;
    bool isDataValid() const override;
    
    // Source file management
    bool hasValidSourceFile() const override;
    QString getSourceFilePath() const override;
    void setSourceFilePath(const QString &path) override;
    bool validateSourceFile(QString &errorMessage) override;
    
    void resetToDefaults() override;
    
    // Settings state capture for preview sync tracking
    QHash<QString, QVariant> getSettingsHash() const override;
    
    // Three-tier UI component access
    QWidget* getSourceFileConfigWidget() override;
    QWidget* getSourceFileSettingsWidget() override;
    QWidget* getOverlaySettingsWidget() override;

private slots:
    void onBrowseButtonClicked();
    void onFilePathChanged();
    void onConvolutionEnabledToggled(bool enabled);
    void onLineshapeTypeChanged(int index);
    void onConvolutionSettingsChanged();
    void onAutoRangeClicked();
    void onSaveRangeOnlyToggled(bool enabled);
    
    // Background operation handlers
    void onConvolutionOperationStarted(const QString &operationId);
    void onConvolutionOperationProgress(const QString &operationId, int percentage, const QString &message);
    void onConvolutionOperationCompleted(const QString &operationId, std::shared_ptr<OverlayBase> result);
    void onConvolutionOperationFailed(const QString &operationId, const QString &error);
    void onConvolutionOperationCancelled(const QString &operationId);

protected:
    // OverlayTypeSpecificWidget interface
    void setupUI() override;
    void setupConnections() override;
    void loadSettings() override;
    void saveSettings() override;

private:
    // Three-tier UI organization
    QWidget *p_sourceFileConfigWidget;
    QWidget *p_sourceFileSettingsWidget; 
    QWidget *p_overlaySettingsWidget;
    
    // Source File Configuration tier (File selection)
    QGroupBox *p_fileSelectionGroup;
    QLineEdit *p_filePathLineEdit;
    QToolButton *p_browseButton;
    QLabel *p_formatLabel;
    QLabel *p_moleculeLabel;
    QLabel *p_transitionCountLabel;
    QLabel *p_frequencyRangeLabel;
    
    // Source File Settings tier (Source-dependent controls)
    QGroupBox *p_sourceFileGroup;
    QCheckBox *p_saveRangeOnlyCheckBox;
    QPushButton *p_autoRangeButton;
    
    // Overlay Settings tier (Convolution - source-independent)
    QGroupBox *p_convolutionGroup;
    QCheckBox *p_convolutionEnabledCheckBox;
    QComboBox *p_lineshapeComboBox;
    QDoubleSpinBox *p_linewidthSpinBox;
    QDoubleSpinBox *p_minFreqSpinBox;
    QDoubleSpinBox *p_maxFreqSpinBox;
    QDoubleSpinBox *p_pointSpacingSpinBox;
    
    // State management
    CatalogData d_catalogData;
    QString d_filePath;
    bool d_fileValid;
    bool d_hasFtData;
    double d_ftYMax;
    
    // Helper methods
    void setupFileSelectionUI();
    void setupSourceFileSettingsUI();
    void setupConvolutionSettingsUI();
    void loadCatalogFile(const QString &filePath);
    void updateFileInfo();
    void updateConvolutionControls();
    void autoSetFrequencyRange();
    void calculateDefaultYScale();
    bool validateConvolutionSettings(QString &errorMessage) const;
    QString formatFrequencyRange(double min, double max) const;
    void configureSpinBox(QDoubleSpinBox *spinBox, const QString &minKey, const QString &maxKey, 
                         const QString &decimalsKey, const QString &stepKey, 
                         double defaultMin, double defaultMax, int defaultDecimals, double defaultStep);
    
    // Background processing support
    void triggerBackgroundConvolution();
    void cancelPendingConvolution();
    
    // Background operation tracking
    QString d_currentConvolutionId;
    bool d_convolutionInProgress;
    
    // Default values
    static constexpr bool DEFAULT_CONVOLUTION_ENABLED = false;
    static constexpr int DEFAULT_LINESHAPE_TYPE = 0; // Lorentzian
    static constexpr double DEFAULT_LINEWIDTH = 100.0;    // kHz FWHM
    static constexpr double DEFAULT_MIN_FREQ = 0.0;       // MHz
    static constexpr double DEFAULT_MAX_FREQ = 1000.0;    // MHz
    static constexpr double DEFAULT_POINT_SPACING = 0.01; // MHz
    static constexpr bool DEFAULT_SAVE_RANGE_ONLY = true;
};

#endif // CATALOGOVERLAYWIDGET_H