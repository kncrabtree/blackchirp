#include "wizardloscanconfigpage.h"

#include <QGroupBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QSettings>
#include <QLabel>
#include <QHBoxLayout>
#include <QFormLayout>

WizardLoScanConfigPage::WizardLoScanConfigPage(QWidget *parent) : ExperimentWizardPage(parent)
{
    setTitle(QString("Configure LO Scan"));
    setSubTitle(QString("Hover over the various fields for more information."));

    p_upBox = new QGroupBox("Upconversion LO");

    p_upStartBox = new QDoubleSpinBox;
    p_upStartBox->setDecimals(6);
    p_upStartBox->setSuffix(QString(" MHz"));
    p_upStartBox->setSingleStep(1000.0);
    p_upStartBox->setToolTip(QString("Starting major step LO frequency."));

    p_upEndBox = new QDoubleSpinBox;
    p_upEndBox->setDecimals(6);
    p_upEndBox->setSuffix(QString(" MHz"));
    p_upEndBox->setSingleStep(1000.0);
    p_upEndBox->setToolTip(QString("Ending LO frequency (including major and minor steps).\nThis is a limit; the frequency will not exceed this value.\nChanging this value will update the major step size."));

    p_upNumMinorBox = new QSpinBox;
    p_upNumMinorBox->setRange(1,10);
    p_upNumMinorBox->setToolTip(QString("Number of minor (small) steps to take per major step.\nThe sign is determined automatically."));

    p_upMinorStepBox = new QDoubleSpinBox;
    p_upMinorStepBox->setDecimals(6);
    p_upMinorStepBox->setSuffix(QString(" MHz"));
    p_upMinorStepBox->setSingleStep(1.0);
    p_upMinorStepBox->setToolTip(QString("Minor step size, if number of minor steps > 1.\nMay change the number of major steps."));

    p_upNumMajorBox = new QSpinBox;
    p_upNumMajorBox->setRange(2,100000);
    p_upNumMajorBox->setToolTip(QString("Number of major steps desired.\nChanging this will update the major step size."));

    p_upMajorStepBox = new QDoubleSpinBox;
    p_upMajorStepBox->setDecimals(6);
    p_upMajorStepBox->setSuffix(QString(" MHz"));
    p_upMajorStepBox->setSingleStep(100.0);
    p_upMajorStepBox->setToolTip(QString("Desired major step size.\nChanging this will update the number of major steps."));

    auto *upgl = new QGridLayout;
    upgl->addWidget(new QLabel("Start"),0,0);
    upgl->addWidget(p_upStartBox,0,1);
    upgl->addWidget(new QLabel("End"),0,2);
    upgl->addWidget(p_upEndBox,0,3);

    upgl->addWidget(new QLabel("Minor Steps/pt"),1,0);
    upgl->addWidget(p_upNumMinorBox,1,1);
    upgl->addWidget(new QLabel("Size"),1,2);
    upgl->addWidget(p_upMinorStepBox,1,3);

    upgl->addWidget(new QLabel("Major Steps"),2,0);
    upgl->addWidget(p_upNumMajorBox,2,1);
    upgl->addWidget(new QLabel("Size"),2,2);
    upgl->addWidget(p_upMajorStepBox,2,3);

    p_upBox->setLayout(upgl);



    p_downBox = new QGroupBox("Upconversion LO");

    p_downStartBox = new QDoubleSpinBox;
    p_downStartBox->setDecimals(6);
    p_downStartBox->setSuffix(QString(" MHz"));
    p_downStartBox->setSingleStep(1000.0);
    p_downStartBox->setToolTip(QString("Starting major step LO frequency."));

    p_downEndBox = new QDoubleSpinBox;
    p_downEndBox->setDecimals(6);
    p_downEndBox->setSuffix(QString(" MHz"));
    p_downEndBox->setSingleStep(1000.0);
    p_downEndBox->setToolTip(QString("Ending LO frequency (including major and minor steps).\nThis is a limit; the frequency will not exceed this value.\nChanging this value will update the major step size."));

    p_downNumMinorBox = new QSpinBox;
    p_downNumMinorBox->setRange(1,10);
    p_downNumMinorBox->setToolTip(QString("Number of minor (small) steps to take per major step.\nThe sign is determined automatically."));

    p_downMinorStepBox = new QDoubleSpinBox;
    p_downMinorStepBox->setDecimals(6);
    p_downMinorStepBox->setSuffix(QString(" MHz"));
    p_downMinorStepBox->setSingleStep(1.0);
    p_downMinorStepBox->setToolTip(QString("Minor step size, if number of minor steps > 1.\nMay change the number of major steps."));

    p_downNumMajorBox = new QSpinBox;
    p_downNumMajorBox->setRange(2,100000);
    p_downNumMajorBox->setToolTip(QString("Number of major steps desired.\nChanging this will update the major step size."));

    p_downMajorStepBox = new QDoubleSpinBox;
    p_downMajorStepBox->setDecimals(6);
    p_downMajorStepBox->setSuffix(QString(" MHz"));
    p_downMajorStepBox->setSingleStep(100.0);
    p_downMajorStepBox->setToolTip(QString("Desired major step size.\nChanging this will update the number of major steps."));

    auto *downgl = new QGridLayout;
    downgl->addWidget(new QLabel("Start"),0,0);
    downgl->addWidget(p_downStartBox,0,1);
    downgl->addWidget(new QLabel("End"),0,2);
    downgl->addWidget(p_downEndBox,0,3);

    downgl->addWidget(new QLabel("Minor Steps/pt"),1,0);
    downgl->addWidget(p_downNumMinorBox,1,1);
    downgl->addWidget(new QLabel("Size"),1,2);
    downgl->addWidget(p_downMinorStepBox,1,3);

    downgl->addWidget(new QLabel("Major Steps"),2,0);
    downgl->addWidget(p_downNumMajorBox,2,1);
    downgl->addWidget(new QLabel("Size"),2,2);
    downgl->addWidget(p_downMajorStepBox,2,3);

    p_downBox->setLayout(downgl);

    auto *otherBox = new QGroupBox(QString("Scan Settings"));
    auto *fl = new QFormLayout;

    p_shotsPerStepBox = new QSpinBox;
    p_shotsPerStepBox->setRange(1,__INT_MAX__);
    p_shotsPerStepBox->setSingleStep(1000);
    p_shotsPerStepBox->setToolTip(QString("Number of shots to acquire at each step (major and minor)."));
    fl->addRow(QString("Shots/Point"),p_shotsPerStepBox);


    p_targetSweepsBox = new QSpinBox;
    p_targetSweepsBox->setRange(1,__INT_MAX__);
    p_targetSweepsBox->setToolTip(QString("Number of sweeps through the total LO range.\nExperiment will end when this number is reached."));
    fl->addRow(QString("Target Sweeps"),p_targetSweepsBox);

    auto *hbl = new QHBoxLayout;
    hbl->addWidget(otherBox);
    hbl->addWidget(p_upBox);
    hbl->addWidget(p_downBox);

    setLayout(hbl);

}

void WizardLoScanConfigPage::initializePage()
{
}

bool WizardLoScanConfigPage::validatePage()
{
    return true;
}

bool WizardLoScanConfigPage::isComplete() const
{
    return true;
}

int WizardLoScanConfigPage::nextId() const
{
    return ExperimentWizard::ChirpConfigPage;
}
