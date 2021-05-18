#ifndef WIZARDDRSCANCONFIGPAGE_H
#define WIZARDDRSCANCONFIGPAGE_H

#include "experimentwizardpage.h"

class QDoubleSpinBox;
class QSpinBox;

class WizardDrScanConfigPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardDrScanConfigPage(QWidget *parent=nullptr);

    // QWizardPage interface
public:
    void initializePage() override;
    bool validatePage() override;
    bool isComplete() const override;
    int nextId() const override;

public slots:
    void updateEndBox();

protected:
    void saveToSettings() const;
    void loadFromSettings();

private:
    QDoubleSpinBox *p_startBox, *p_stepSizeBox, *p_endBox;
    QSpinBox *p_numStepsBox, *p_shotsBox;

    RfConfig d_rfConfig;

};

#endif // WIZARDDRSCANCONFIGPAGE_H
