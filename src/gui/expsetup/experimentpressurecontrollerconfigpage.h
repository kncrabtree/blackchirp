#ifndef EXPERIMENTPRESSURECONTROLLERCONFIGPAGE_H
#define EXPERIMENTPRESSURECONTROLLERCONFIGPAGE_H

#include "experimentconfigpage.h"

class PressureControlWidget;

namespace BC::Key::WizPC {
inline constexpr QLatin1StringView key{"WizardPressureControllerConfigPage"};
}

class ExperimentPressureControllerConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentPressureControllerConfigPage(const QString hwKey, const QString title, Experiment *exp, QWidget *parent = nullptr);

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;

private:
    PressureControlWidget *p_pcw;
};

#endif // EXPERIMENTPRESSURECONTROLLERCONFIGPAGE_H
