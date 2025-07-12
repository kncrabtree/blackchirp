#ifndef BCEXPOVERLAYDIALOG_H
#define BCEXPOVERLAYDIALOG_H

#include <QSpinBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QToolButton>
#include <QFormLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QLabel>
#include <memory>

#include <data/analysis/ft.h>
#include <data/experiment/overlaytypes.h>
#include "overlayconfigdialog.h"

class ExperimentViewWidget;
class FtmwViewWidget;

class BCExpOverlayDialog : public OverlayConfigDialog
{
    Q_OBJECT

public:
    explicit BCExpOverlayDialog(FtmwViewWidget *parent);
    ~BCExpOverlayDialog();

    // Template Method pattern interface
    std::shared_ptr<OverlayBase> createTypeSpecificOverlay() const override;
    void configureTypeSpecificOverlay(std::shared_ptr<OverlayBase> overlay) const override;

protected:
    // OverlayConfigDialog interface
    void setupTypeSpecificUI() override;
    void setupTypeSpecificConnections() override;
    void initializeTypeSpecificDefaults() override;
    bool validateTypeSpecificSettings(QString &errorMessage) override;
    bool isTypeSpecificDataValid() const override;

private slots:
    void onExperimentNumberChanged(int number);
    void onUsePathToggled(bool enabled);
    void onBrowseButtonClicked();
    void onPathChanged();
    void validateExperiment();
    void onConfigureFtClicked();

private:
    // UI elements - Experiment selection
    QSpinBox *p_experimentNumberSpinBox;
    QCheckBox *p_usePathCheckBox;
    QLineEdit *p_pathLineEdit;
    QToolButton *p_browseButton;
    
    // FT Configuration
    QPushButton *p_configureFtButton;
    
    // State
    bool d_experimentValid;
    Ft d_configuredFt;
    bool d_hasFtData;
    
    // Helper methods
    void setupExperimentSelection();
    void setupFtConfiguration();
    void resetFtConfiguration();
    QString getExperimentPath() const;
    bool validateExperimentPath(const QString &path, QString &errorMessage);
};

#endif // BCEXPOVERLAYDIALOG_H
