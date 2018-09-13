#include "wizardvalidationpage.h"

#include <QVBoxLayout>
#include <QLabel>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QCheckBox>
#include <QToolButton>

#include "experimentwizard.h"
#include "ioboardconfigmodel.h"
#include "validationmodel.h"

WizardValidationPage::WizardValidationPage(QWidget *parent) :
    ExperimentWizardPage(parent)
{
    setTitle(QString("Validation Settings"));
    setSubTitle(QString("Configure IO board channels and set up conditions that will automatically abort the experiment."));

    QHBoxLayout *hbl = new QHBoxLayout;

    QGroupBox *iobox = new QGroupBox(QString("IO Board"));
    QVBoxLayout *iol = new QVBoxLayout;

    QLabel *an = new QLabel(QString("Analog Channels"));
    an->setAlignment(Qt::AlignCenter);
    iol->addWidget(an,0,Qt::AlignCenter);

    p_analogView = new QTableView();
    IOBoardConfigModel *anmodel = new IOBoardConfigModel(QString("AIN"),p_analogView);
    p_analogView->setModel(anmodel);
    p_analogView->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    p_analogView->horizontalHeader()->setSectionResizeMode(2,QHeaderView::Stretch);
    p_analogView->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
    iol->addWidget(p_analogView,1,Qt::AlignCenter);

    QLabel *di = new QLabel(QString("Digital Channels"));
    di->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    iol->addWidget(di,0,Qt::AlignCenter);

    p_digitalView = new QTableView();
    IOBoardConfigModel *dmodel = new IOBoardConfigModel(QString("DIN"),p_digitalView);
    p_digitalView->setModel(dmodel);
    p_digitalView->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    p_digitalView->horizontalHeader()->setSectionResizeMode(2,QHeaderView::Stretch);
    p_digitalView->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
    iol->addWidget(p_digitalView,1,Qt::AlignCenter);

    iobox->setLayout(iol);
    hbl->addWidget(iobox,1,Qt::AlignCenter);

    QVBoxLayout *vl = new QVBoxLayout;
    vl->addStretch(1);

    QLabel *val = new QLabel(QString("Validation"));
    val->setAlignment(Qt::AlignCenter);
    vl->addWidget(val,0,Qt::AlignCenter);

    p_validationView = new QTableView;
    ValidationModel *valmodel = new ValidationModel(p_validationView);
    p_validationView->setModel(valmodel);
    p_validationView->setItemDelegateForColumn(0,new CompleterLineEditDelegate);
    p_validationView->setItemDelegateForColumn(1,new ValidationDoubleSpinBoxDelegate);
    p_validationView->setItemDelegateForColumn(2,new ValidationDoubleSpinBoxDelegate);
    p_validationView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    p_validationView->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
    vl->addWidget(p_validationView,1,Qt::AlignCenter);

    QHBoxLayout *tl = new QHBoxLayout;
    tl->addStretch(1);

    p_addButton = new QToolButton();
    p_addButton->setIcon(QIcon(QString(":/icons/add.png")));
    p_addButton->setText(QString(""));
    tl->addWidget(p_addButton,0);

    p_removeButton = new QToolButton();
    p_removeButton->setIcon(QIcon(QString(":/icons/remove.png")));
    p_removeButton->setText(QString(""));
    tl->addWidget(p_removeButton,0);
    tl->addStretch(1);
    vl->addLayout(tl);

    vl->addStretch(1);

    hbl->addLayout(vl,1);

    setLayout(hbl);

    connect(p_addButton,&QToolButton::clicked,[=](){ valmodel->addNewItem(); });
    connect(p_removeButton,&QToolButton::clicked,[=](){
        QModelIndexList l = p_validationView->selectionModel()->selectedIndexes();
        if(l.isEmpty())
            return;

        QList<int> rowList;
        for(int i=0; i<l.size(); i++)
        {
            if(!rowList.contains(l.at(i).row()))
                rowList.append(l.at(i).row());
        }

        std::stable_sort(rowList.begin(),rowList.end());

        for(int i=rowList.size()-1; i>=0; i--)
            valmodel->removeRows(rowList.at(i),1,QModelIndex());
    });
}



int WizardValidationPage::nextId() const
{
    return ExperimentWizard::SummaryPage;
}

IOBoardConfig WizardValidationPage::getConfig() const
{
    IOBoardConfig out;
    out.setAnalogChannels(static_cast<IOBoardConfigModel*>(p_analogView->model())->getConfig());
    out.setDigitalChannels(static_cast<IOBoardConfigModel*>(p_digitalView->model())->getConfig());
    return out;
}

QMap<QString, BlackChirp::ValidationItem> WizardValidationPage::getValidation() const
{
    auto l = static_cast<ValidationModel*>(p_validationView->model())->getList();
    QMap<QString,BlackChirp::ValidationItem> out;
    for(int i=0; i<l.size(); i++)
        out.insert(l.at(i).key,l.at(i));
    return out;
}


void WizardValidationPage::initializePage()
{
    auto e = getExperiment();
    auto c = e.iobConfig();

    dynamic_cast<IOBoardConfigModel*>(p_analogView->model())->setFromConfig(c);
    dynamic_cast<IOBoardConfigModel*>(p_digitalView->model())->setFromConfig(c);
    dynamic_cast<ValidationModel*>(p_validationView->model())->setFromMap(e.validationItems());

    p_analogView->resizeColumnsToContents();
    p_digitalView->resizeColumnsToContents();
    p_validationView->resizeColumnsToContents();

}

bool WizardValidationPage::validatePage()
{
    auto e = getExperiment();

    IOBoardConfig c;
    c.setAnalogChannels(static_cast<IOBoardConfigModel*>(p_analogView->model())->getConfig());
    c.setDigitalChannels(static_cast<IOBoardConfigModel*>(p_digitalView->model())->getConfig());
    e.setIOBoardConfig(c);

    auto l = static_cast<ValidationModel*>(p_validationView->model())->getList();
    e.setValidationItems(QMap<QString, BlackChirp::ValidationItem>());
    for(int i=0; i<l.size(); i++)
        e.addValidationItem(l.at(i));

    emit experimentUpdate(e);
    return true;
}
