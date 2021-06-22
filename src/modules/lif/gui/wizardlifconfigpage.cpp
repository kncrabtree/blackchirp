#include "wizardlifconfigpage.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QSettings>
#include <QApplication>
#include <QGroupBox>
#include <QLabel>
#include <QComboBox>

#include <src/modules/lif/gui/lifcontrolwidget.h>
#include <src/modules/lif/gui/liflasercontroldoublespinbox.h>
#include <src/gui/wizard/experimentwizard.h>

WizardLifConfigPage::WizardLifConfigPage(QWidget *parent) :
    ExperimentWizardPage(BC::Key::WizLif::key,parent)
{
    setTitle(QString("LIF Configuration"));
    setSubTitle(QString("Configure the parameters for the LIF Acquisition. Oscilloscope settings are immediately applied. Integration gates and shots per point can be set by right-clicking the plot."));

    auto *vbl = new QVBoxLayout;

    p_lifControl = new LifControlWidget(this);
    connect(this,&WizardLifConfigPage::newTrace,p_lifControl,&LifControlWidget::newTrace);
    //connect signals/slots

    vbl->addWidget(p_lifControl,1);

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    auto *optionsBox = new QGroupBox(QString("Acquisition Options"));
    auto *ofl = new QFormLayout;

    p_orderBox = new QComboBox(this);
    p_orderBox->addItem(QString("Frequency First"),QVariant::fromValue(BlackChirp::LifOrderFrequencyFirst));
    p_orderBox->addItem(QString("Delay First"),QVariant::fromValue(BlackChirp::LifOrderDelayFirst));
    p_orderBox->setToolTip(QString("Controls the order in which the delay and laser frequency will be changed during the scan.\n\nFrequency first: Acquire spectrum at single delay point, then increment delay and repeat.\nDelay first: Acquire time trace at a single frequency, then increment frequency and repeat.\n\nNote that the order is irrelevant if either the delay or frequency is set to a single point."));
    auto *lbl = new QLabel(QString("Scan order"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    ofl->addRow(lbl,p_orderBox);

    p_completeBox = new QComboBox(this);
    p_completeBox->addItem(QString("Stop integrating when all points acquired."),
                           QVariant::fromValue(BlackChirp::LifStopWhenComplete));
    p_completeBox->addItem(QString("Continue integrating until entire experiment is complete."),
                           QVariant::fromValue(BlackChirp::LifContinueUntilExperimentComplete));
    p_completeBox->setToolTip(QString("Configures behavior if LIF scan finishes before the rest of the experiment.\n\nStop integrating: Once all points are acquired, no more shots will be integrated.\nContinue: Scan will return to beginning and continue integrating data until remainder of experiment is completed or aborted.\n\n This setting is not applicable if LIF is the only measurement being made or if other parts of the experiment finish before LIF."));
    lbl = new QLabel(QString("Completion Behavior"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    ofl->addRow(lbl,p_completeBox);

    optionsBox->setLayout(ofl);
    vbl->addWidget(optionsBox,0);

    auto *hbl = new QHBoxLayout;

    auto *delayBox = new QGroupBox(QString("LIF Delay"));
    auto *delayFl = new QFormLayout;

    p_delaySingle = new QCheckBox(this);
    p_delaySingle->setChecked(s.value(QString("lifConfig/delaySingle"),false).toBool());
    p_delaySingle->setToolTip(QString("If checked, the LIF delay will not change during the scan, and will remain at the value in the start box."));
    lbl = new QLabel(QString("Single Point"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    delayFl->addRow(lbl,p_delaySingle);

    p_delayStart = new QDoubleSpinBox(this);
    p_delayStart->setRange(0.0,100000.0);
    p_delayStart->setDecimals(3);
    p_delayStart->setSuffix(QString::fromUtf16(u" µs"));
    p_delayStart->setSingleStep(10.0);
    p_delayStart->setToolTip(QString("Starting delay for LIF measurement. For a single delay scan, this will be the value used."));
    lbl = new QLabel(QString("Start"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    delayFl->addRow(lbl,p_delayStart);


    p_delayStep = new QDoubleSpinBox(this);
    p_delayStep->setRange(-100.0,100.0);
    p_delayStep->setDecimals(3);
    p_delayStep->setSingleStep(1.0);
    p_delayStep->setSuffix(QString::fromUtf16(u" µs"));
    p_delayStep->setToolTip(QString("Step size between delay points."));
    lbl = new QLabel(QString("Step Size"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    delayFl->addRow(lbl,p_delayStep);

    p_delayNum = new QSpinBox(this);
    p_delayNum->setRange(1,1000);
    p_delayNum->setToolTip(QString("Number of delay points. This will be set to 1 automatically if the single delay scan is checked."));
    lbl = new QLabel(QString("Points"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    delayFl->addRow(lbl,p_delayNum);

    QPushButton *delayButton = new QPushButton(QString("Set to Start"),this);
    delayFl->addWidget(delayButton);

    delayBox->setLayout(delayFl);

    hbl->addWidget(delayBox,1);


    auto *laserBox = new QGroupBox(QString("LIF Laser"));
    auto *laserFl = new QFormLayout;

    p_laserSingle = new QCheckBox(this);
    p_laserSingle->setChecked(s.value(QString("lifConfig/laserSingle"),false).toBool());
    p_laserSingle->setToolTip(QString("If checked, the LIF laser frequency will not change during the scan, and will remain at the value in the start box."));
    lbl = new QLabel(QString("Single Point"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    laserFl->addRow(lbl,p_laserSingle);

    p_laserStart = new LifLaserControlDoubleSpinBox(this);
    p_laserStart->configure();
    p_laserStart->setToolTip(QString("Starting position for LIF measurement. For a single frequency scan, this will be the value used."));
    lbl = new QLabel(QString("Start"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    laserFl->addRow(lbl,p_laserStart);



    p_laserStep = new LifLaserControlDoubleSpinBox(this);
    p_laserStep->configure(true);
    p_laserStep->setToolTip(QString("Step size between laser position points."));
    lbl = new QLabel(QString("Step Size"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    laserFl->addRow(lbl,p_laserStep);

    p_laserNum = new QSpinBox(this);
    p_laserNum->setRange(1,1000);
    p_laserNum->setToolTip(QString("Number of laser points. this will be set to 1 automatically if the single frequency box is checked."));
    lbl = new QLabel(QString("Points"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    laserFl->addRow(lbl,p_laserNum);

    QPushButton *laserButton = new QPushButton(QString("Set to Start"),this);
    laserFl->addWidget(laserButton);
    connect(laserButton,&QPushButton::clicked,[=](){
       emit laserPosUpdate(p_laserStart->value());
    });

    laserBox->setLayout(laserFl);

    hbl->addWidget(laserBox,1);

    vbl->addLayout(hbl,0);

    setLayout(vbl);

    connect(p_delaySingle,&QCheckBox::toggled,[=](bool en){
        if(en)
        {
            p_delayNum->setEnabled(false);
            p_delayStep->setEnabled(false);
            p_delayNum->setValue(1);
        }
        else
        {
            p_delayNum->setEnabled(true);
            p_delayStep->setEnabled(true);
        }
    });

    connect(p_laserSingle,&QCheckBox::toggled,[=](bool en){
        if(en)
        {
            p_laserNum->setEnabled(false);
            p_laserStep->setEnabled(false);
            p_laserNum->setValue(1);
        }
        else
        {
            p_laserNum->setEnabled(true);
            p_laserStep->setEnabled(true);
        }
    });

    connect(this,&WizardLifConfigPage::scopeConfigChanged,p_lifControl,&LifControlWidget::scopeConfigChanged);
    connect(p_lifControl,&LifControlWidget::updateScope,this,&WizardLifConfigPage::updateScope);
    connect(p_lifControl,&LifControlWidget::laserPosUpdate,this,&WizardLifConfigPage::laserPosUpdate);

    registerField(QString("delayStart"),p_delayStart,"value","valueChanged");
}

WizardLifConfigPage::~WizardLifConfigPage()
= default;

void WizardLifConfigPage::setFromConfig(const LifConfig c)
{
    p_orderBox->setCurrentIndex(p_orderBox->findData(QVariant::fromValue(c.order())));
    p_completeBox->setCurrentIndex(p_completeBox->findData(QVariant::fromValue(c.completeMode())));
    p_delaySingle->setChecked(c.numDelayPoints() == 1);
    p_delayStart->setValue(c.delayRange().first);
    p_delayNum->setValue(c.numDelayPoints());
    p_delayStep->setValue(c.delayStep());
    p_laserSingle->setChecked(c.numLaserPoints() == 1);
    p_laserStart->setValue(c.laserRange().first);
    p_laserNum->setValue(c.numLaserPoints());
    p_laserStep->setValue(c.laserStep());

    //Control widget is set on the fly
}

void WizardLifConfigPage::setLaserPos(const double pos)
{
    p_lifControl->setLaserPos(pos);
}

void WizardLifConfigPage::initializePage()
{
    auto e = getExperiment();
    setFromConfig(e->lifConfig());
}

bool WizardLifConfigPage::validatePage()
{
    auto e = getExperiment();
    LifConfig out;
    out = p_lifControl->getSettings(out);
    out.setCompleteMode(p_completeBox->currentData().value<BlackChirp::LifCompleteMode>());
    out.setOrder(p_orderBox->currentData().value<BlackChirp::LifScanOrder>());
    out.setDelayParameters(p_delayStart->value(),p_delayStep->value(),p_delayNum->value());
    out.setLaserParameters(p_laserStart->value(),p_laserStep->value(),p_laserNum->value());


    out.setEnabled();
    e->setLifConfig(out);
    
    return true;

}

int WizardLifConfigPage::nextId() const
{
    auto e = getExperiment();
    if(e->ftmwConfig().isEnabled())
        return ExperimentWizard::RfConfigPage;

    return ExperimentWizard::PulseConfigPage;
}
