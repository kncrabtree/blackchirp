#ifndef EXPERIMENTLIFCONFIGPAGE_H
#define EXPERIMENTLIFCONFIGPAGE_H

#include <gui/expsetup/experimentconfigpage.h>

class LifControlWidget;

namespace BC::Key::WizLif {
static const QString key{"WizardLifConfigPage"};
static const QString title{"LIF Configuration"};
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
