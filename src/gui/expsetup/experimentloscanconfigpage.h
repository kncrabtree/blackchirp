#ifndef EXPERIMENTLOSCANCONFIGPAGE_H
#define EXPERIMENTLOSCANCONFIGPAGE_H

#include "experimentconfigpage.h"
#include "loscanconfigwidget.h"

class ExperimentLOScanConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentLOScanConfigPage(Experiment *exp, QWidget *parent = nullptr);

public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;

private:
    LOScanConfigWidget *p_widget;
};

#endif // EXPERIMENTLOSCANCONFIGPAGE_H
