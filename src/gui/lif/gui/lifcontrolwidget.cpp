#include <gui/lif/gui/lifcontrolwidget.h>

#include <gui/widget/digitizerconfigwidget.h>
#include <gui/lif/gui/liftraceplot.h>
#include <gui/lif/gui/liflaserwidget.h>
#include <gui/lif/gui/lifprocessingwidget.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QGroupBox>
#include <QToolButton>
#include <QLabel>
#include <QSpinBox>

#include <gui/style/themecolors.h>

using namespace Qt::StringLiterals;

LifControlWidget::LifControlWidget(const QString& digitizerHwKey, const QString& laserHwKey, QWidget *parent) :
    QWidget(parent), SettingsStorage(BC::Key::LifControl::key),
    ps_cfg(std::make_shared<LifConfig>(digitizerHwKey)), d_laserHwKey(laserHwKey)
{
    initializeWidget();
}

void LifControlWidget::initializeWidget()
{
    auto vbl = new QVBoxLayout;

    p_lifTracePlot = new LifTracePlot(this);
    p_lifTracePlot->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);


    auto hbl = new QHBoxLayout;
    auto dgb = new QGroupBox("LIF Digitizer",this);
    dgb->setSizePolicy(QSizePolicy::Preferred,QSizePolicy::Minimum);

    auto vbl2 = new QVBoxLayout;

    auto dl = new QLabel("Analog Channel 1 is for the LIF signal. Channel 2, if enabled, is the reference channel.",this);
    dl->setWordWrap(true);
    vbl2->addWidget(dl);

    p_digWidget = new DigitizerConfigWidget(BC::Key::LifControl::lifDigWidget,
                                            ps_cfg->digitizerConfig().headerKey(),false,dgb);
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

    // Compact icon-only controls; tooltips carry the meaning.
    p_resetButton = new QToolButton(this);
    p_resetButton->setIcon(ThemeColors::createThemedIcon(":/icons/arrow-path.svg",ThemeColors::IconSecondary,this));
    p_resetButton->setToolTip("Reset the averaged trace."_L1);
    connect(p_resetButton,&QToolButton::clicked,p_lifTracePlot,&LifTracePlot::reset);
    hbl2->addWidget(p_resetButton);

    hbl2->addSpacerItem(new QSpacerItem(1,1));

    p_startAcqButton = new QToolButton(this);
    p_startAcqButton->setIcon(ThemeColors::createThemedIcon(":/icons/play.svg",ThemeColors::IconPrimary,this));
    p_startAcqButton->setToolTip("Start Acquisition"_L1);

    p_stopAcqButton = new QToolButton(this);
    p_stopAcqButton->setIcon(ThemeColors::createThemedIcon(":/icons/stop.svg",ThemeColors::IconPrimary,this));
    p_stopAcqButton->setToolTip("Stop Acquisition"_L1);

    p_stopAcqButton->setEnabled(false);
    hbl2->addWidget(p_startAcqButton);
    hbl2->addWidget(p_stopAcqButton);
    vbl2->addLayout(hbl2,0);

    dgb->setLayout(vbl2);
    hbl->addWidget(dgb,1);

    auto rightvbl = new QVBoxLayout;

    auto lgb = new QGroupBox("Laser",this);
    lgb->setSizePolicy(QSizePolicy::Preferred,QSizePolicy::Minimum);
    auto vbl3 = new QVBoxLayout;
    p_laserWidget = new LifLaserWidget(d_laserHwKey, lgb);
    vbl3->addWidget(p_laserWidget);
    lgb->setLayout(vbl3);
    rightvbl->addWidget(lgb,0);

    auto pgb = new QGroupBox("Processing",this);
    pgb->setSizePolicy(QSizePolicy::Preferred,QSizePolicy::Minimum);
    auto pvbl = new QVBoxLayout;
    p_procWidget = new LifProcessingWidget(true,pgb);
    pvbl->addWidget(p_procWidget);
    pgb->setLayout(pvbl);
    rightvbl->addWidget(pgb,1);
    p_procWidget->setEnabled(false);


    hbl->addLayout(rightvbl,1);

    // Plot over controls in a splitter so the plot can be dragged
    // larger/smaller; it gets the larger initial share and the bottom
    // stays at its compact size hint.
    auto bottomWidget = new QWidget;
    bottomWidget->setLayout(hbl);

    auto splitter = new QSplitter(Qt::Vertical,this);
    splitter->addWidget(p_lifTracePlot);
    splitter->addWidget(bottomWidget);
    splitter->setStretchFactor(0,1);
    splitter->setStretchFactor(1,0);
    splitter->setChildrenCollapsible(false);
    splitter->setSizes({600,250});

    vbl->addWidget(splitter);
    setLayout(vbl);

    // Width floor so the digitizer / data-transfer / trigger boxes are
    // not clipped; height is left to the layout so the dialog is no
    // longer pinned tall (the splitter balances plot vs. controls).
    setMinimumWidth(1000);

    connect(p_startAcqButton,&QToolButton::clicked,this,&LifControlWidget::startAcquisition);
    connect(p_stopAcqButton,&QToolButton::clicked,this,&LifControlWidget::stopAcquisition);

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

    auto &cfg = ps_cfg->digitizerConfig();
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
    emit startSignal(*ps_cfg);
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
        LifTrace l(ps_cfg->digitizerConfig(),b,0,0,8);
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
    p_digWidget->setFromConfig(cfg.digitizerConfig());
    p_avgBox->setValue(cfg.d_shotsPerPoint);
    p_procWidget->setAll(cfg.d_procSettings);
    *ps_cfg = cfg;
}

void LifControlWidget::toConfig(LifConfig &cfg)
{
    p_digWidget->toConfig(cfg.digitizerConfig());
    cfg.d_shotsPerPoint = p_avgBox->value();
    cfg.d_procSettings = p_procWidget->getSettings();
    auto it = cfg.digitizerConfig().d_analogChannels.find(cfg.digitizerConfig().d_refChannel);
    if(it != cfg.digitizerConfig().d_analogChannels.end())
        cfg.digitizerConfig().d_refEnabled = it->second.enabled;
    else
        cfg.digitizerConfig().d_refEnabled = false;
}
