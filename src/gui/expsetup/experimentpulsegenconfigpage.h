#ifndef EXPERIMENTPULSEGENCONFIGPAGE_H
#define EXPERIMENTPULSEGENCONFIGPAGE_H

#include "experimentconfigpage.h"

class PulseConfigWidget;

namespace BC::Key::WizPulse {
static const QString key{"WizardPulsePage"};
}

class ExperimentPulseGenConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentPulseGenConfigPage(const QString hwKey, const QString title, Experiment *exp, QWidget *parent = nullptr);

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;

private:
    PulseConfigWidget *p_pcw;
};

#endif // EXPERIMENTPULSEGENCONFIGPAGE_H
