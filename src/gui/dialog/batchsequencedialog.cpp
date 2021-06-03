#include "batchsequencedialog.h"
#include "ui_batchsequencedialog.h"

#include <QSettings>

BatchSequenceDialog::BatchSequenceDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::BatchSequenceDialog)
{
    ui->setupUi(this);

    connect(ui->cancelButton,&QPushButton::clicked,this,&BatchSequenceDialog::reject);
    connect(ui->quickButton,&QPushButton::clicked,this,[=](){ done(d_quickCode); });
    connect(ui->configureButton,&QPushButton::clicked,this,[=](){ done(d_configureCode); });

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lastBatchSequence"));

    ui->numberOfExperimentsSpinBox->setValue(s.value(QString("numExpts"),1).toInt());
    ui->timeBetweenExperimentsSpinBox->setValue(s.value(QString("interval"),300).toInt());
    ui->automaticallyExportCheckBox->setChecked(s.value(QString("autoExport"),false).toBool());

    s.endGroup();
    s.sync();
}

BatchSequenceDialog::~BatchSequenceDialog()
{
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

bool BatchSequenceDialog::autoExport() const
{
    return ui->automaticallyExportCheckBox->isChecked();
}

void BatchSequenceDialog::saveToSettings() const
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lastBatchSequence"));

    s.setValue(QString("numExpts"),numExperiments());
    s.setValue(QString("interval"),interval());
    s.setValue(QString("autoExport"),autoExport());

    s.endGroup();
    s.sync();
}
