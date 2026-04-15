#ifndef EXPERIMENTLIFCONFIGPAGE_H
#define EXPERIMENTLIFCONFIGPAGE_H

#include <gui/expsetup/experimentconfigpage.h>

class LifControlWidget;

namespace BC::Key::WizLif {
inline constexpr QLatin1StringView key{"WizardLifConfigPage"};
inline constexpr QLatin1StringView title{"LIF Configuration"};
}

class ExperimentLifConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentLifConfigPage(Experiment *exp, QWidget *parent = nullptr);

    LifControlWidget *lifControlWidget() { return p_lcw; }

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;

private:
    LifControlWidget *p_lcw;
};

#endif // EXPERIMENTLIFCONFIGPAGE_H
