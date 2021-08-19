#include "batchsequencedialog.h"

using namespace BC::Key::SeqDialog;

BatchSequenceDialog::BatchSequenceDialog(QWidget *parent) :
    QDialog(parent), SettingsStorage(key),
    ui(new Ui::BatchSequenceDialog)
{
    ui->setupUi(this);

    connect(ui->cancelButton,&QPushButton::clicked,this,&BatchSequenceDialog::reject);
    connect(ui->quickButton,&QPushButton::clicked,this,[=](){ done(quickCode); });
    connect(ui->configureButton,&QPushButton::clicked,this,[=](){ done(configureCode); });

    ui->numberOfExperimentsSpinBox->setValue(get<int>(batchExperiments,1));
    ui->timeBetweenExperimentsSpinBox->setValue(get<int>(batchInterval,300));

    registerGetter(batchExperiments,ui->numberOfExperimentsSpinBox,&QSpinBox::value);
    registerGetter(batchInterval,ui->timeBetweenExperimentsSpinBox,&QSpinBox::value);
}

BatchSequenceDialog::~BatchSequenceDialog()
{
    clearGetters(false);
    delete ui;
}

void BatchSequenceDialog::setQuickExptEnabled(bool en)
{
    ui->quickButton->setEnabled(en);
    ui->configureButton->setDefault(!en);
    ui->quickButton->setDefault(en);
}

int BatchSequenceDialog::numExperiments() const
{
    return ui->numberOfExperimentsSpinBox->value();
}

int BatchSequenceDialog::interval() const
{
    return ui->timeBetweenExperimentsSpinBox->value();
}
