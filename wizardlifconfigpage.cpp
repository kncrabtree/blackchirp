#include "wizardlifconfigpage.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QSettings>
#include <QApplication>
#include <QGroupBox>
#include <QLabel>
#include <QComboBox>

#include "lifcontrolwidget.h"
#include "experimentwizard.h"

WizardLifConfigPage::WizardLifConfigPage(QWidget *parent) :
    QWizardPage(parent)
{
    setTitle(QString("LIF Configuration"));
    setSubTitle(QString("Configure the parameters for the LIF Acquisition. Oscilloscope settings are immediately applied. Integration gates and shots per point can be set by right-clicking the plot."));

    QVBoxLayout *vbl = new QVBoxLayout;

    p_lifControl = new LifControlWidget(this);
    connect(this,&WizardLifConfigPage::newTrace,p_lifControl,&LifControlWidget::newTrace);
    //connect signals/slots

    vbl->addWidget(p_lifControl,1);

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    QGroupBox *optionsBox = new QGroupBox(QString("Acquisition Options"));
    QFormLayout *ofl = new QFormLayout;

    p_orderBox = new QComboBox(this);
    p_orderBox->addItem(QString("Frequency First"),QVariant::fromValue(BlackChirp::LifOrderFrequencyFirst));
    p_orderBox->addItem(QString("Delay First"),QVariant::fromValue(BlackChirp::LifOrderDelayFirst));
    p_orderBox->setCurrentIndex(s.value(QString("lifConfig/scanOrder"),0).toInt());
    p_orderBox->setToolTip(QString("Controls the order in which the delay and laser frequency will be changed during the scan.\n\nFrequency first: Acquire spectrum at single delay point, then increment delay and repeat.\nDelay first: Acquire time trace at a single frequency, then increment frequency and repeat.\n\nNote that the order is irrelevant if either the delay or frequency is set to a single point."));
    ofl->addRow(QString("Scan order"),p_orderBox);

    p_completeBox = new QComboBox(this);
    p_completeBox->addItem(QString("Stop integrating when all points acquired."),
                           QVariant::fromValue(BlackChirp::LifStopWhenComplete));
    p_completeBox->addItem(QString("Continue integrating until entire experiment is complete."),
                           QVariant::fromValue(BlackChirp::LifContinueUntilExperimentComplete));
    p_completeBox->setCurrentIndex(s.value(QString("lifConfig/completeMode"),1).toInt());
    p_completeBox->setToolTip(QString("Configures behavior if LIF scan finishes before the rest of the experiment.\n\nStop integrating: Once all points are acquired, no more shots will be integrated.\nContinue: Scan will return to beginning and continue integrating data until remainder of experiment is completed or aborted.\n\n This setting is not applicable if LIF is the only measurement being made or if other parts of the experiment finish before LIF."));
    ofl->addRow(QString("Completion behavior"),p_completeBox);

    optionsBox->setLayout(ofl);
    vbl->addWidget(optionsBox,0);

    QHBoxLayout *hbl = new QHBoxLayout;

    QGroupBox *delayBox = new QGroupBox(QString("LIF Delay"));
    QFormLayout *delayFl = new QFormLayout;

    p_delaySingle = new QCheckBox(this);
    p_delaySingle->setChecked(s.value(QString("lifConfig/delaySingle"),false).toBool());
    p_delaySingle->setToolTip(QString("If checked, the LIF delay will not change during the scan, and will remain at the value in the start box."));
    delayFl->addRow(QString("Single Point"),p_delaySingle);

    p_delayStart = new QDoubleSpinBox(this);
    p_delayStart->setRange(0.0,100000.0);
    p_delayStart->setDecimals(3);
    p_delayStart->setValue(s.value(QString("lifConfig/delayStart"),1000.0).toDouble());
    p_delayStart->setSuffix(QString::fromUtf16(u" µs"));
    p_delayStart->setSingleStep(10.0);
    p_delayStart->setToolTip(QString("Starting delay for LIF measurement. For a single delay scan, this will be the value used."));
    delayFl->addRow(QString("Start"),p_delayStart);

    p_delayEnd = new QDoubleSpinBox(this);
    p_delayEnd->setRange(0.0,100000.0);
    p_delayEnd->setDecimals(3);
    p_delayEnd->setValue(s.value(QString("lifConfig/delayEnd"),1100.0).toDouble());
    p_delayEnd->setSuffix(QString::fromUtf16(u" µs"));
    p_delayEnd->setSingleStep(10.0);
    p_delayEnd->setToolTip(QString("Ending delay for LIF measurement. May be greater or less than starting delay, and need not be an integral number of steps from start.\nIf |end-start| < step, the delay will remain at the starting value as if the single point box were checked."));
    delayFl->addRow(QString("End"),p_delayEnd);

    p_delayStep = new QDoubleSpinBox(this);
    p_delayStep->setRange(0.05,100.0);
    p_delayStep->setDecimals(3);
    p_delayStep->setValue(s.value(QString("lifConfig/delayStep"),1.0).toDouble());
    p_delayStep->setSingleStep(1.0);
    p_delayStep->setSuffix(QString::fromUtf16(u" µs"));
    p_delayStep->setToolTip(QString("Step size between delay points."));
    delayFl->addRow(QString("Step Size"),p_delayStep);

    QPushButton *delayButton = new QPushButton(QString("Set to Start"),this);
    delayFl->addWidget(delayButton);

    delayBox->setLayout(delayFl);

    hbl->addWidget(delayBox,1);


    QGroupBox *laserBox = new QGroupBox(QString("LIF Laser"));
    QFormLayout *laserFl = new QFormLayout;

    p_laserSingle = new QCheckBox(this);
    p_laserSingle->setChecked(s.value(QString("lifConfig/laserSingle"),false).toBool());
    p_laserSingle->setToolTip(QString("If checked, the LIF laser frequency will not change during the scan, and will remain at the value in the start box."));
    laserFl->addRow(QString("Single Point"),p_laserSingle);

    s.beginGroup(QString("lifLaser"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    p_laserStart = new QDoubleSpinBox(this);
    p_laserStart->setRange(s.value(QString("minFreq"),10000.0).toDouble(),
                           s.value(QString("maxFreq"),100000.0).toDouble());
    p_laserStart->setDecimals(2);
    p_laserStart->setValue(s.value(QString("lifConfig/laserStart"),20000.0).toDouble());
    p_laserStart->setSuffix(QString::fromUtf16(u" cm⁻¹"));
    p_laserStart->setSingleStep(100.0);
    p_laserStart->setToolTip(QString("Starting frequency for LIF measurement. For a single frequency scan, this will be the value used."));
    laserFl->addRow(QString("Start"),p_laserStart);


    p_laserEnd = new QDoubleSpinBox(this);
    p_laserEnd->setRange(s.value(QString("minFreq"),10000.0).toDouble(),
                         s.value(QString("maxFreq"),100000.0).toDouble());
    p_laserEnd->setDecimals(2);
    p_laserEnd->setValue(s.value(QString("lifConfig/laserEnd"),20100.0).toDouble());
    p_laserEnd->setSuffix(QString::fromUtf16(u" cm⁻¹"));
    p_laserEnd->setSingleStep(100.0);
    p_laserEnd->setToolTip(QString("Ending laser frequency for LIF measurement. May be greater or less than starting frequency, and need not be an integral number of steps from start.\nIf |end-start| < step, the frequency will remain at the starting value as if the single point box were checked."));
    laserFl->addRow(QString("End"),p_laserEnd);
    s.endGroup();
    s.endGroup();

    p_laserStep = new QDoubleSpinBox(this);
    p_laserStep->setRange(0.01,100.0);
    p_laserStep->setDecimals(2);
    p_laserStep->setValue(s.value(QString("lifConfig/laserStep"),1.0).toDouble());
    p_laserStep->setSingleStep(1.0);
    p_laserStep->setSuffix(QString::fromUtf16(u" cm⁻¹"));
    p_laserStep->setToolTip(QString("Step size between frequency points."));
    laserFl->addRow(QString("Step Size"),p_laserStep);

    QPushButton *laserButton = new QPushButton(QString("Set to Start"),this);
    laserFl->addWidget(laserButton);

    laserBox->setLayout(laserFl);

    hbl->addWidget(laserBox,1);

    vbl->addLayout(hbl,0);

    setLayout(vbl);

    connect(p_delaySingle,&QCheckBox::toggled,[=](bool en){
        if(en)
        {
            p_delayEnd->setEnabled(false);
            p_delayStep->setEnabled(false);
            p_delayEnd->setValue(p_delayStart->value());
        }
        else
        {
            p_delayEnd->setEnabled(true);
            p_delayStep->setEnabled(true);
        }
    });

    connect(p_laserSingle,&QCheckBox::toggled,[=](bool en){
        if(en)
        {
            p_laserEnd->setEnabled(false);
            p_laserStep->setEnabled(false);
            p_laserEnd->setValue(p_laserStart->value());
        }
        else
        {
            p_laserEnd->setEnabled(true);
            p_laserStep->setEnabled(true);
        }
    });

    connect(this,&WizardLifConfigPage::scopeConfigChanged,p_lifControl,&LifControlWidget::scopeConfigChanged);
    connect(p_lifControl,&LifControlWidget::updateScope,this,&WizardLifConfigPage::updateScope);
    connect(p_lifControl,&LifControlWidget::lifColorChanged,this,&WizardLifConfigPage::lifColorChanged);

    registerField(QString("delayStart"),p_delayStart,"value","valueChanged");
}

WizardLifConfigPage::~WizardLifConfigPage()
{

}

void WizardLifConfigPage::saveToSettings() const
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    //scope settings and graph settings are saved on the fly, so don't worry about those
    s.beginGroup(QString("lifConfig"));
    s.setValue(QString("scanOrder"),p_orderBox->currentIndex());
    s.setValue(QString("completeMode"),p_completeBox->currentIndex());
    s.setValue(QString("delaySingle"),p_delaySingle->isChecked());
    s.setValue(QString("delayStart"),p_delayStart->value());
    s.setValue(QString("delayEnd"),p_delayEnd->value());
    s.setValue(QString("delayStep"),p_delayStep->value());
    s.setValue(QString("laserSingle"),p_laserSingle->isChecked());
    s.setValue(QString("laserStart"),p_laserStart->value());
    s.setValue(QString("laserEnd"),p_laserEnd->value());
    s.setValue(QString("laserStep"),p_laserStep->value());

    s.endGroup();
    s.sync();
}

LifConfig WizardLifConfigPage::getConfig()
{
    LifConfig out;
    out = p_lifControl->getSettings(out);
    out.setCompleteMode(p_completeBox->currentData().value<BlackChirp::LifCompleteMode>());
    out.setOrder(p_orderBox->currentData().value<BlackChirp::LifScanOrder>());
    out.setDelayParameters(p_delayStart->value(),p_delayEnd->value(),p_delayStep->value());
    out.setFrequencyParameters(p_laserStart->value(),p_laserEnd->value(),p_laserStep->value());

    out.validate();
    return out;
}



void WizardLifConfigPage::initializePage()
{
}

bool WizardLifConfigPage::validatePage()
{
    return getConfig().isValid();
}

int WizardLifConfigPage::nextId() const
{
    if(field(QString("ftmw")).toBool())
        return ExperimentWizard::ChirpConfigPage;
    else
        return ExperimentWizard::PulseConfigPage;
}
