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

#include "lifcontrolwidget.h"
#include "experimentwizard.h"

WizardLifConfigPage::WizardLifConfigPage(QWidget *parent) :
    QWizardPage(parent)
{
    setTitle(QString("LIF Configuration"));
    setSubTitle(QString("Configure the parameters for the LIF Acquisition. Any changes you make to the oscilloscope settings will be applied immediately; you can preview the results in the plot. Right-click the plot to set integration ranges and number of shots to acquire per step. For the laser frequency and delay controls, if you check the single point checkbox, the starting value will be used for the entire scan, and the stop value will be set equal to the start. Note that if you leave the checkbox unchecked, but leave the stop value equal to the start, the scan will still be a single point scan. You can use the buttons to set the laser or LIF delay to the value in the start box."));

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
    delayFl->addRow(QString("Single Point"),p_delaySingle);

    p_delayStart = new QDoubleSpinBox(this);
    p_delayStart->setRange(0.0,100000.0);
    p_delayStart->setDecimals(3);
    p_delayStart->setValue(s.value(QString("lifConfig/delayStart"),1000.0).toDouble());
    p_delayStart->setSuffix(QString::fromUtf16(u" µs"));
    p_delayStart->setSingleStep(10.0);
    delayFl->addRow(QString("Start"),p_delayStart);

    p_delayEnd = new QDoubleSpinBox(this);
    p_delayEnd->setRange(0.0,100000.0);
    p_delayEnd->setDecimals(3);
    p_delayEnd->setValue(s.value(QString("lifConfig/delayEnd"),1100.0).toDouble());
    p_delayEnd->setSuffix(QString::fromUtf16(u" µs"));
    p_delayEnd->setSingleStep(10.0);
    delayFl->addRow(QString("End"),p_delayEnd);

    p_delayStep = new QDoubleSpinBox(this);
    p_delayStep->setRange(0.05,100.0);
    p_delayStep->setDecimals(3);
    p_delayStep->setValue(s.value(QString("lifConfig/delayStep"),1.0).toDouble());
    p_delayStep->setSingleStep(1.0);
    p_delayStep->setSuffix(QString::fromUtf16(u" µs"));
    delayFl->addRow(QString("Step Size"),p_delayStep);

    QPushButton *delayButton = new QPushButton(QString("Set to Start"),this);
    delayFl->addWidget(delayButton);

    delayBox->setLayout(delayFl);

    hbl->addWidget(delayBox,1);


    QGroupBox *laserBox = new QGroupBox(QString("LIF laser"));
    QFormLayout *laserFl = new QFormLayout;

    p_laserSingle = new QCheckBox(this);
    p_laserSingle->setChecked(s.value(QString("lifConfig/laserSingle"),false).toBool());
    laserFl->addRow(QString("Single Point"),p_laserSingle);

    p_laserStart = new QDoubleSpinBox(this);
    p_laserStart->setRange(s.value(QString("lifLaser/minFreq"),10000.0).toDouble(),
                           s.value(QString("lifLaser/maxFreq"),100000.0).toDouble());
    p_laserStart->setDecimals(2);
    p_laserStart->setValue(s.value(QString("lifConfig/laserStart"),20000.0).toDouble());
    p_laserStart->setSuffix(QString::fromUtf16(u" µs"));
    p_laserStart->setSingleStep(100.0);
    laserFl->addRow(QString("Start"),p_laserStart);

    p_laserEnd = new QDoubleSpinBox(this);
    p_laserEnd->setRange(s.value(QString("lifLaser/minFreq"),10000.0).toDouble(),
                         s.value(QString("lifLaser/maxFreq"),100000.0).toDouble());
    p_laserEnd->setDecimals(2);
    p_laserEnd->setValue(s.value(QString("lifConfig/laserEnd"),20100.0).toDouble());
    p_laserEnd->setSuffix(QString::fromUtf16(u" µs"));
    p_laserEnd->setSingleStep(100.0);
    laserFl->addRow(QString("End"),p_laserEnd);

    p_laserStep = new QDoubleSpinBox(this);
    p_laserStep->setRange(0.01,100.0);
    p_laserStep->setDecimals(2);
    p_laserStep->setValue(s.value(QString("lifConfig/laserStep"),1.0).toDouble());
    p_laserStep->setSingleStep(1.0);
    p_laserStep->setSuffix(QString::fromUtf16(u" µs"));
    laserFl->addRow(QString("Step Size"),p_laserStep);

    QPushButton *laserButton = new QPushButton(QString("Set to Start"),this);
    laserFl->addWidget(laserButton);

    laserBox->setLayout(laserFl);

    hbl->addWidget(laserBox,1);

    vbl->addLayout(hbl,0);

    setLayout(vbl);

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
