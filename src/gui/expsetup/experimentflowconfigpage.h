#ifndef EXPERIMENTFLOWCONFIGPAGE_H
#define EXPERIMENTFLOWCONFIGPAGE_H

#include "experimentconfigpage.h"

class GasControlWidget;

namespace BC::Key::WizFlow {
static const QString key{"WizardFlowConfigPage"};
}

class ExperimentFlowConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentFlowConfigPage(const QString hwKey, const QString title, Experiment *exp, QWidget *parent = nullptr);

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;

private:
    GasControlWidget *p_gcw;
};

#endif // EXPERIMENTFLOWCONFIGPAGE_H
