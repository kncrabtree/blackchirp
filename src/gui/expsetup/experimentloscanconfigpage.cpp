#include "experimentloscanconfigpage.h"

#include <QVBoxLayout>

ExperimentLOScanConfigPage::ExperimentLOScanConfigPage(Experiment *exp, QWidget *parent)
    : ExperimentConfigPage(BC::Key::WizLoScan::key, BC::Key::WizLoScan::title, exp, parent)
{
    p_widget = new LOScanConfigWidget(exp, this);
    auto *l = new QVBoxLayout;
    l->setContentsMargins(0, 0, 0, 0);
    l->addWidget(p_widget);
    setLayout(l);
    connect(p_widget, &LOScanConfigWidget::error, this, &ExperimentLOScanConfigPage::error);
    connect(p_widget, &LOScanConfigWidget::warning, this, &ExperimentLOScanConfigPage::warning);
}

void ExperimentLOScanConfigPage::initialize() { p_widget->initialize(); }
bool ExperimentLOScanConfigPage::validate() { return p_widget->validate(); }
void ExperimentLOScanConfigPage::apply() { p_widget->apply(); }
