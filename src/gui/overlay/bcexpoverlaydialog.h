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

#include <data/analysis/ft.h>
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
    void onConfigureFtClicked();
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
    
    // FT Configuration
    QPushButton *p_configureFtButton;
    
    QDialogButtonBox *p_buttonBox;
    
    // State
    QString d_currentExperimentPath;
    bool d_experimentValid;
    QStringList d_plotNames;
    Ft d_configuredFt;
    bool d_hasFtData;

    ExperimentViewWidget *p_msw;
    
    // Helper methods
    void setupUI();
    void setupExperimentSelection();
    void setupFrameSelection();
    void setupOverlayBaseOptions();
    void setupFtConfiguration();
    void setupConnections();
    void initializeDefaults();
    void updateValidationStatus(bool valid, const QString &message = QString());
    void updateOkButtonState();
    QString getExperimentPath() const;
    bool validateExperimentPath(const QString &path, QString &errorMessage);
};

#endif // BCEXPOVERLAYDIALOG_H
