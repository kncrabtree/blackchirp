#ifndef EXPERIMENTFTMWDIGITIZERCONFIGPAGE_H
#define EXPERIMENTFTMWDIGITIZERCONFIGPAGE_H

#include "experimentconfigpage.h"

class FtmwDigitizerConfigWidget;

namespace BC::Key::WizFtDig {
static const QString key{"WizardFtmwDigitizerPage"};
static const QString title{"Digitizer"};
}

class ExperimentFtmwDigitizerConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentFtmwDigitizerConfigPage(Experiment *exp, QWidget *parent=nullptr);

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;

private:
    FtmwDigitizerConfigWidget *p_dc;
};

#endif // EXPERIMENTFTMWDIGITIZERCONFIGPAGE_H
