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
#include <QSpinBox>

LifControlWidget::LifControlWidget(QWidget *parent) :
    QWidget(parent), SettingsStorage(BC::Key::LifControl::key)
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

    p_digWidget = new DigitizerConfigWidget(BC::Key::LifControl::lifDigWidget,
                                            BC::Key::hwKey(BC::Key::LifDigi::lifScope,0),dgb);
    vbl2->addWidget(p_digWidget);

    auto hbl2 = new QHBoxLayout;
    hbl2->addWidget(new QLabel("Averages"));

    p_avgBox = new QSpinBox;
    p_avgBox->setRange(1,1000000000);
    p_avgBox->setValue(get(BC::Key::LifControl::avgs,10));
    p_avgBox->setSingleStep(10);
    connect(p_avgBox,qOverload<int>(&QSpinBox::valueChanged),p_lifTracePlot,&LifTracePlot::setNumAverages);
    registerGetter(BC::Key::LifControl::avgs,p_avgBox,&QSpinBox::value);
    hbl2->addWidget(p_avgBox);

    p_resetButton = new QPushButton("Reset");
    connect(p_resetButton,&QPushButton::clicked,p_lifTracePlot,&LifTracePlot::reset);
    hbl2->addWidget(p_resetButton);

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
    auto pvbl = new QVBoxLayout;
    p_procWidget = new LifProcessingWidget(true,pgb);
    pvbl->addWidget(p_procWidget);
    pgb->setLayout(pvbl);
    rightvbl->addWidget(pgb,1);
    p_procWidget->setEnabled(false);


    hbl->addLayout(rightvbl,1);

    vbl->addLayout(hbl,1);
    setLayout(vbl);

    connect(p_startAcqButton,&QPushButton::clicked,this,&LifControlWidget::startAcquisition);
    connect(p_stopAcqButton,&QPushButton::clicked,this,&LifControlWidget::stopAcquisition);

    connect(p_procWidget,&LifProcessingWidget::settingChanged,this,[this](){
        p_lifTracePlot->setAllProcSettings(p_procWidget->getSettings());
    });

    connect(p_laserWidget,&LifLaserWidget::changePosition,this,&LifControlWidget::changeLaserPosSignal);
    connect(p_laserWidget,&LifLaserWidget::changeFlashlamp,this,&LifControlWidget::changeLaserFlashlampSignal);

}

LifControlWidget::~LifControlWidget()
{
    emit stopSignal();
}

void LifControlWidget::startAcquisition()
{

    auto &cfg = d_cfg.scopeConfig();
    p_digWidget->toConfig(cfg);
    auto it = cfg.d_analogChannels.find(cfg.d_refChannel);
    if(it != cfg.d_analogChannels.end())
        cfg.d_refEnabled = it->second.enabled;
    else
        cfg.d_refEnabled = false;

    //send cfg parameters to processing widget; disable editing digitizer; configure buttons
    p_procWidget->initialize(cfg.d_recordLength,cfg.d_refEnabled);
    p_procWidget->setEnabled(true);

    p_digWidget->setEnabled(false);
    p_startAcqButton->setEnabled(false);
    p_stopAcqButton->setEnabled(true);

    d_acquiring = true;
    emit startSignal(d_cfg);
}

void LifControlWidget::stopAcquisition()
{
    p_procWidget->setEnabled(false);
    p_digWidget->setEnabled(true);
    p_startAcqButton->setEnabled(true);
    p_stopAcqButton->setEnabled(false);
    d_acquiring = false;

    emit stopSignal();
}

void LifControlWidget::acquisitionStarted()
{
    p_lifTracePlot->reset();
    p_lifTracePlot->setNumAverages(p_avgBox->value());
    p_lifTracePlot->setAllProcSettings(p_procWidget->getSettings());

}

void LifControlWidget::newWaveform(const QVector<qint8> b)
{
    if(d_acquiring)
    {
        //set bitShift to 8 to provide extra bits for rolling average
        LifTrace l(d_cfg.scopeConfig(),b,0,0,8);
        p_lifTracePlot->processTrace(l);
    }
}

void LifControlWidget::setLaserPosition(const double d)
{
    p_laserWidget->setPosition(d);
}

void LifControlWidget::setFlashlamp(bool en)
{
    p_laserWidget->setFlashlamp(en);
}

void LifControlWidget::setFromConfig(const LifConfig &cfg)
{
    p_digWidget->setFromConfig(cfg.scopeConfig());
    p_avgBox->setValue(cfg.d_shotsPerPoint);
    p_procWidget->setAll(cfg.d_procSettings);
    d_cfg = cfg;
}

void LifControlWidget::toConfig(LifConfig &cfg)
{
    p_digWidget->toConfig(cfg.scopeConfig());
    cfg.d_shotsPerPoint = p_avgBox->value();
    cfg.d_procSettings = p_procWidget->getSettings();
    auto it = cfg.scopeConfig().d_analogChannels.find(cfg.scopeConfig().d_refChannel);
    if(it != cfg.scopeConfig().d_analogChannels.end())
        cfg.scopeConfig().d_refEnabled = it->second.enabled;
    else
        cfg.scopeConfig().d_refEnabled = false;
}


QSize LifControlWidget::sizeHint() const
{
    return {1000,800};
}
