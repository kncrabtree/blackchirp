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

    p_laserStart = new QDoubleSpinBox(this);
    p_laserStart->setRange(s.value(QString("lifLaser/minFreq"),10000.0).toDouble(),
                           s.value(QString("lifLaser/maxFreq"),100000.0).toDouble());
    p_laserStart->setDecimals(2);
    p_laserStart->setValue(s.value(QString("lifConfig/laserStart"),20000.0).toDouble());
    p_laserStart->setSuffix(QString::fromUtf16(u" cm⁻¹"));
    p_laserStart->setSingleStep(100.0);
    p_laserStart->setToolTip(QString("Starting frequency for LIF measurement. For a single frequency scan, this will be the value used."));
    laserFl->addRow(QString("Start"),p_laserStart);

    p_laserEnd = new QDoubleSpinBox(this);
    p_laserEnd->setRange(s.value(QString("lifLaser/minFreq"),10000.0).toDouble(),
                         s.value(QString("lifLaser/maxFreq"),100000.0).toDouble());
    p_laserEnd->setDecimals(2);
    p_laserEnd->setValue(s.value(QString("lifConfig/laserEnd"),20100.0).toDouble());
    p_laserEnd->setSuffix(QString::fromUtf16(u" cm⁻¹"));
    p_laserEnd->setSingleStep(100.0);
    p_laserEnd->setToolTip(QString("Ending laser frequency for LIF measurement. May be greater or less than starting frequency, and need not be an integral number of steps from start.\nIf |end-start| < step, the frequency will remain at the starting value as if the single point box were checked."));
    laserFl->addRow(QString("End"),p_laserEnd);

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

}

WizardLifConfigPage::~WizardLifConfigPage()
{

}



void WizardLifConfigPage::initializePage()
{
}

bool WizardLifConfigPage::validatePage()
{
    return true;
}

int WizardLifConfigPage::nextId() const
{
    if(field(QString("ftmw")).toBool())
        return ExperimentWizard::ChirpConfigPage;
    else
        return ExperimentWizard::PulseConfigPage;
}
