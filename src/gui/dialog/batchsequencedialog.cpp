#include "batchsequencedialog.h"
#include "ui_batchsequencedialog.h"

BatchSequenceDialog::BatchSequenceDialog(QWidget *parent) :
    QDialog(parent), SettingsStorage("lastBatchSequence"),
    ui(new Ui::BatchSequenceDialog)
{
    ui->setupUi(this);

    connect(ui->cancelButton,&QPushButton::clicked,this,&BatchSequenceDialog::reject);
    connect(ui->quickButton,&QPushButton::clicked,this,[=](){ done(quickCode); });
    connect(ui->configureButton,&QPushButton::clicked,this,[=](){ done(configureCode); });

    ui->numberOfExperimentsSpinBox->setValue(get<int>(BC::Key::batchExperiments,1));
    ui->timeBetweenExperimentsSpinBox->setValue(get<int>(BC::Key::batchInterval,300));
    ui->automaticallyExportCheckBox->setChecked(get<bool>(BC::Key::batchAutoExport,false));

    registerGetter(BC::Key::batchExperiments,ui->numberOfExperimentsSpinBox,&QSpinBox::value);
    registerGetter(BC::Key::batchInterval,ui->timeBetweenExperimentsSpinBox,&QSpinBox::value);
    registerGetter(BC::Key::batchAutoExport,static_cast<QAbstractButton*>(ui->automaticallyExportCheckBox),&QAbstractButton::isChecked);
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
