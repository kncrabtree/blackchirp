#ifndef EXPERIMENTIOBOARDCONFIGPAGE_H
#define EXPERIMENTIOBOARDCONFIGPAGE_H

#include "experimentconfigpage.h"

class IOBoardConfigWidget;

namespace BC::Key::WizIOB {
static const QString key{"WizardIOBoardConfigPage"};
}

class ExperimentIOBoardConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentIOBoardConfigPage(const QString hwKey, const QString title, Experiment *exp, QWidget *parent = nullptr);

private:
    IOBoardConfig d_config;
    IOBoardConfigWidget *p_iobcw;

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;
};

#endif // EXPERIMENTIOBOARDCONFIGPAGE_H
