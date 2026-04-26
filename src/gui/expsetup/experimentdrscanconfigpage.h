#ifndef EXPERIMENTDRSCANCONFIGPAGE_H
#define EXPERIMENTDRSCANCONFIGPAGE_H

#include "experimentconfigpage.h"
#include "drscanconfigwidget.h"

class ExperimentDRScanConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentDRScanConfigPage(Experiment *exp, QWidget *parent = nullptr);

public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;

private:
    DRScanConfigWidget *p_widget;
};

#endif // EXPERIMENTDRSCANCONFIGPAGE_H
