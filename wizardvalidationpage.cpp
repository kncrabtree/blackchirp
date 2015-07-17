#include "wizardvalidationpage.h"

#include <QVBoxLayout>
#include <QLabel>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QCheckBox>

#include "experimentwizard.h"
#include "ioboardconfigmodel.h"

WizardValidationPage::WizardValidationPage(QWidget *parent) :
    QWizardPage(parent)
{
    setTitle(QString("Validation Settings"));
    setSubTitle(QString("Configure IO board channels and set up conditions that will automatically abort the experiment."));

    QHBoxLayout *hbl = new QHBoxLayout;
    hbl->addSpacerItem(new QSpacerItem(20,0,QSizePolicy::Expanding));

    QGroupBox *iobox = new QGroupBox(QString("IO Board"));
    QVBoxLayout *iol = new QVBoxLayout;

    QLabel *an = new QLabel(QString("Analog Channels"));
    an->setAlignment(Qt::AlignCenter);
    iol->addWidget(an,0,Qt::AlignCenter);

    p_analogView = new QTableView();
    IOBoardConfigModel *anmodel = new IOBoardConfigModel(d_config.analogList(),d_config.numAnalogChannels(),d_config.reservedAnalogChannels(),QString("AIN"),p_analogView);
    p_analogView->setModel(anmodel);
    p_analogView->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    p_analogView->horizontalHeader()->setSectionResizeMode(2,QHeaderView::Stretch);
    p_analogView->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
    iol->addWidget(p_analogView,1,Qt::AlignCenter);

    QLabel *di = new QLabel(QString("Digital Channels"));
    di->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    iol->addWidget(di,0,Qt::AlignCenter);

    p_digitalView = new QTableView();
    IOBoardConfigModel *dmodel = new IOBoardConfigModel(d_config.digitalList(),d_config.numDigitalChannels(),d_config.reservedDigitalChannels(),QString("DIN"),p_digitalView);
    p_digitalView->setModel(dmodel);
    p_digitalView->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    p_digitalView->horizontalHeader()->setSectionResizeMode(2,QHeaderView::Stretch);
    p_digitalView->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
    iol->addWidget(p_digitalView,1,Qt::AlignCenter);

    iobox->setLayout(iol);
    hbl->addWidget(iobox,1,Qt::AlignCenter);

    QVBoxLayout *vl = new QVBoxLayout;
    QLabel *val = new QLabel(QString("Validation"));
    val->setAlignment(Qt::AlignCenter);
    vl->addWidget(val,0,Qt::AlignCenter);

    p_validationView = new QTableView;
    //create validation model
    p_validationView->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
    vl->addWidget(p_validationView,1,Qt::AlignCenter);

    hbl->addLayout(vl,1);
    hbl->addSpacerItem(new QSpacerItem(20,0,QSizePolicy::Expanding));

    setLayout(hbl);
}



int WizardValidationPage::nextId() const
{
    return ExperimentWizard::SummaryPage;
}
