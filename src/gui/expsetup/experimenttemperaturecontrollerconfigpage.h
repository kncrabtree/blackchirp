#ifndef EXPERIMENTTEMPERATURECONTROLLERCONFIGPAGE_H
#define EXPERIMENTTEMPERATURECONTROLLERCONFIGPAGE_H

#include "experimentconfigpage.h"

class TemperatureControlWidget;

namespace BC::Key::WizTC {
static const QString key{"WizardTemperatureControllerConfigPage"};
}

class ExperimentTemperatureControllerConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentTemperatureControllerConfigPage(const QString hwKey, const QString title, Experiment *exp, QWidget *parent = nullptr);

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;

private:
    TemperatureControlWidget *p_tcw;
};

#endif // EXPERIMENTTEMPERATURECONTROLLERCONFIGPAGE_H
