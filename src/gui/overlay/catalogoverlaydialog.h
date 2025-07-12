#ifndef CATALOGOVERLAYDIALOG_H
#define CATALOGOVERLAYDIALOG_H

#include <QSpinBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QToolButton>
#include <QFormLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <memory>

#include <data/experiment/overlaytypes.h>
#include <data/experiment/catalogdata.h>
#include <data/experiment/catalogparserregistry.h>
#include <data/storage/settingsstorage.h>
#include "overlayconfigdialog.h"

class FtmwViewWidget;

namespace BC::Key::CatalogDialog {
static const QString key{"CatalogOverlayDialog"};
static const QString geometry{"geometry"};
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
 * @brief Dialog for creating and configuring catalog overlays
 * 
 * Provides UI for selecting catalog files (SPCAT, XIAM, etc.) with automatic
 * format detection and configuring convolution settings for realistic spectrum
 * comparison. Supports multiple catalog formats through the parser registry.
 * 
 * Uses SettingsStorage for persistent configuration of convolution settings
 * and metasettings for spinbox behavior customization.
 */
class CatalogOverlayDialog : public OverlayConfigDialog, public SettingsStorage
{
    Q_OBJECT

public:
    explicit CatalogOverlayDialog(FtmwViewWidget *parent);
    ~CatalogOverlayDialog();

    // Template Method pattern interface
    std::shared_ptr<OverlayBase> createTypeSpecificOverlay() const override;
    void configureTypeSpecificOverlay(std::shared_ptr<OverlayBase> overlay) const override;

public slots:
    void accept() override;
    void reject() override;

protected:
    // OverlayConfigDialog interface
    void setupTypeSpecificUI() override;
    void setupTypeSpecificConnections() override;
    void initializeTypeSpecificDefaults() override;
    bool validateTypeSpecificSettings(QString &errorMessage) override;
    bool isTypeSpecificDataValid() const override;
    
    // Qt overrides
    void closeEvent(QCloseEvent *event) override;

private slots:
    void onBrowseButtonClicked();
    void onFilePathChanged();
    void onConvolutionEnabledToggled(bool enabled);
    void onLineshapeTypeChanged(int index);
    void onConvolutionSettingsChanged();
    void onAutoRangeClicked();

private:
    // UI elements - File selection
    QLineEdit *p_filePathLineEdit;
    QToolButton *p_browseButton;
    QLabel *p_formatLabel;
    QLabel *p_moleculeLabel;
    QLabel *p_transitionCountLabel;
    QLabel *p_frequencyRangeLabel;
    
    // UI elements - Convolution settings
    QCheckBox *p_convolutionEnabledCheckBox;
    QComboBox *p_lineshapeComboBox;
    QDoubleSpinBox *p_linewidthSpinBox;
    QDoubleSpinBox *p_minFreqSpinBox;
    QDoubleSpinBox *p_maxFreqSpinBox;
    QDoubleSpinBox *p_pointSpacingSpinBox;
    QPushButton *p_autoRangeButton;
    QCheckBox *p_saveRangeOnlyCheckBox;
    
    // State
    CatalogData d_catalogData;
    QString d_filePath;
    bool d_fileValid;
    bool d_hasFtData;
    double d_ftYMax;
    
    // Helper methods
    void setupFileSelection();
    void setupConvolutionSettings();
    void loadCatalogFile(const QString &filePath);
    void updateFileInfo();
    void updateConvolutionControls();
    void autoSetFrequencyRange();
    void calculateDefaultYScale();
    bool validateConvolutionSettings(QString &errorMessage);
    QString formatFrequencyRange(double min, double max) const;
    
    // Settings management
    void loadSettings();
    void saveSettings();
    void configureSpinBox(QDoubleSpinBox *spinBox, const QString &minKey, const QString &maxKey, 
                         const QString &decimalsKey, const QString &stepKey, 
                         double defaultMin, double defaultMax, int defaultDecimals, double defaultStep);
    
    // Default values
    static constexpr bool DEFAULT_CONVOLUTION_ENABLED = false;
    static constexpr int DEFAULT_LINESHAPE_TYPE = 0; // Lorentzian
    static constexpr double DEFAULT_LINEWIDTH = 100.0;    // kHz FWHM
    static constexpr double DEFAULT_MIN_FREQ = 0.0;       // MHz
    static constexpr double DEFAULT_MAX_FREQ = 1000.0;    // MHz
    static constexpr double DEFAULT_POINT_SPACING = 0.01; // MHz
    static constexpr bool DEFAULT_SAVE_RANGE_ONLY = true;
};

#endif // CATALOGOVERLAYDIALOG_H