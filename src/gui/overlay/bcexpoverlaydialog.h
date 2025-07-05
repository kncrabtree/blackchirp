#ifndef BCEXPOVERLAYDIALOG_H
#define BCEXPOVERLAYDIALOG_H

#include <QDialog>
#include <QSpinBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QToolButton>
#include <QRadioButton>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <memory>

#include <data/analysis/ftworker.h>
#include <data/experiment/overlaytypes.h>
#include "overlaybaseoptionswidget.h"

class ExperimentViewWidget;
class FtmwViewWidget;

class BCExpOverlayDialog : public QDialog
{
    Q_OBJECT

public:
    explicit BCExpOverlayDialog(const QStringList &plotNames, FtmwViewWidget *parent = nullptr);
    ~BCExpOverlayDialog();

    std::shared_ptr<OverlayBase> createOverlay() const;

private slots:
    void onExperimentNumberChanged(int number);
    void onUsePathToggled(bool enabled);
    void onBrowseButtonClicked();
    void onPathChanged();
    void validateExperiment();
    void onProcessingSettingsChanged();
    void onManualSettingsClicked();
    void onDialogAccepted();

public slots:
    void accept() override;

private:
    // Parent widget for accessing current settings
    FtmwViewWidget *p_ftmwViewWidget;
    
    // UI elements - Experiment selection
    QSpinBox *p_experimentNumberSpinBox;
    QCheckBox *p_usePathCheckBox;
    QLineEdit *p_pathLineEdit;
    QToolButton *p_browseButton;
    QLabel *p_validationLabel;
    
    // Frame selection
    QSpinBox *p_frameSpinBox;
    
    // Overlay base options
    OverlayBaseOptionsWidget *p_overlayOptionsWidget;
    
    // Processing settings options
    QRadioButton *p_useExperimentSettingsRadio;
    QRadioButton *p_useCurrentSettingsRadio;
    QRadioButton *p_useManualSettingsRadio;
    QPushButton *p_manualSettingsButton;
    QLabel *p_settingsStatusLabel;
    
    QDialogButtonBox *p_buttonBox;
    
    // State
    QString d_currentExperimentPath;
    bool d_experimentValid;
    bool d_hasExperimentSettings;
    FtWorker::FidProcessingSettings d_experimentSettings;
    FtWorker::FidProcessingSettings d_currentSettings;
    FtWorker::FidProcessingSettings d_manualSettings;
    bool d_hasManualSettings;
    QStringList d_plotNames;

    ExperimentViewWidget *p_msw;
    
    // Helper methods
    void setupUI();
    void setupExperimentSelection();
    void setupFrameSelection();
    void setupOverlayBaseOptions();
    void setupProcessingSettings();
    void setupConnections();
    void initializeDefaults();
    void updateValidationStatus(bool valid, const QString &message = QString());
    QString getExperimentPath() const;
    bool validateExperimentPath(const QString &path, QString &errorMessage);
    void updateProcessingSettingsOptions();
    void updateSettingsStatus();
    FtWorker::FidProcessingSettings getSelectedProcessingSettings() const;
    void getCurrentSettingsFromParent();
};

#endif // BCEXPOVERLAYDIALOG_H
