#include "modules/lif/hardware/lifdigitizer/lifscope.h"
#include <modules/lif/gui/lifcontrolwidget.h>

#include <gui/widget/digitizerconfigwidget.h>
#include <modules/lif/gui/liftraceplot.h>
#include <modules/lif/gui/liflaserwidget.h>
#include <modules/lif/gui/lifprocessingwidget.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QLabel>

LifControlWidget::LifControlWidget(QWidget *parent) :
    QWidget(parent)
{
    auto vbl = new QVBoxLayout;

    p_lifTracePlot = new LifTracePlot(this);
    vbl->addWidget(p_lifTracePlot,1);


    auto hbl = new QHBoxLayout;
    auto dgb = new QGroupBox("LIF Digitizer",this);

    auto vbl2 = new QVBoxLayout;

    auto dl = new QLabel("Analog Channel 1 is for the LIF signal. Channel 2, if enabled, is the reference channel.",this);
    dl->setWordWrap(true);
    vbl2->addWidget(dl);

    p_digWidget = new DigitizerConfigWidget(BC::Key::LifControl::lifDigWidget,BC::Key::LifDigi::lifScope,dgb);
    vbl2->addWidget(p_digWidget);

    auto hbl2 = new QHBoxLayout;
    hbl2->addSpacerItem(new QSpacerItem(1,1));

    p_startAcqButton = new QPushButton("Start Acquisition",this);
    p_stopAcqButton = new QPushButton("Stop Acquisition",this);

    p_stopAcqButton->setEnabled(false);
    hbl2->addWidget(p_startAcqButton);
    hbl2->addWidget(p_stopAcqButton);
    vbl2->addLayout(hbl2,0);

    dgb->setLayout(vbl2);
    hbl->addWidget(dgb,1);

    auto rightvbl = new QVBoxLayout;

    auto lgb = new QGroupBox("Laser",this);
    auto vbl3 = new QVBoxLayout;
    p_laserWidget = new LifLaserWidget(lgb);
    vbl3->addWidget(p_laserWidget);
    lgb->setLayout(vbl3);
    rightvbl->addWidget(lgb,0);

    auto pgb = new QGroupBox("Processing",this);
    p_procWidget = new LifProcessingWidget(true,pgb);
    pgb->setLayout(p_procWidget->layout());
    rightvbl->addWidget(pgb,1);
    pgb->setEnabled(false);


    hbl->addLayout(rightvbl,1);


    vbl->addLayout(hbl,1);



    setLayout(vbl);
}

LifControlWidget::~LifControlWidget()
{
}


QSize LifControlWidget::sizeHint() const
{
    return {1000,800};
}
