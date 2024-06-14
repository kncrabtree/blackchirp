#include "experimentioboardconfigpage.h"

using namespace BC::Key::WizIOB;

#include <QVBoxLayout>
#include <gui/widget/ioboardconfigwidget.h>

ExperimentIOBoardConfigPage::ExperimentIOBoardConfigPage(const QString hwKey, const QString title, Experiment *exp, QWidget *parent) :
    ExperimentConfigPage(key,title,exp,parent), d_config(hwKey)
{
    auto vbl = new QVBoxLayout;

    auto ic = exp->getOptHwConfig<IOBoardConfig>(hwKey);
    d_config = *ic.lock();
    p_iobcw = new IOBoardConfigWidget(d_config,this);
    vbl->addWidget(p_iobcw);
    // vbl->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Minimum,QSizePolicy::MinimumExpanding));

    setLayout(vbl);
}


void ExperimentIOBoardConfigPage::initialize()
{
}

bool ExperimentIOBoardConfigPage::validate()
{
    return true;
}

void ExperimentIOBoardConfigPage::apply()
{
    p_iobcw->toConfig(d_config);
    p_exp->addOptHwConfig(d_config);
}
