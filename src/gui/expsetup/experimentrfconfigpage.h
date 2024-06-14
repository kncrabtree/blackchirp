#ifndef EXPERIMENTRFCONFIGPAGE_H
#define EXPERIMENTRFCONFIGPAGE_H

#include "experimentconfigpage.h"

class RfConfigWidget;

namespace BC::Key::WizRf {
static const QString key{"WizardRfConfigPage"};
static const QString title{"RF Configuration"};
}

class ExperimentRfConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentRfConfigPage(Experiment *exp, const QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks, QWidget *parent = nullptr);

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;

private:
    RfConfigWidget *p_rfc;
    QHash<RfConfig::ClockType, RfConfig::ClockFreq> d_clocks;
};

#endif // EXPERIMENTRFCONFIGPAGE_H
