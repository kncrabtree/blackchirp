#include "exportbatchdialog.h"
#include "ui_exportbatchdialog.h"

#include <QFileDialog>

#include <QSettings>
#include <QMessageBox>

#include "analysis.h"
#include "experiment.h"

ExportBatchDialog::ExportBatchDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ExportBatchDialog)
{
    ui->setupUi(this);


    QString path = BlackChirp::getExportDir();
    ui->pathLineEdit->setText(path);

    connect(ui->selectButton,&QPushButton::clicked,this,&ExportBatchDialog::selectDirectory);
    connect(ui->rangeLineEdit,&QLineEdit::textChanged,this,&ExportBatchDialog::checkComplete);
    ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
}

ExportBatchDialog::~ExportBatchDialog()
{
    delete ui;
}

void ExportBatchDialog::selectDirectory()
{
    QString path = QFileDialog::getExistingDirectory(this,QString("Choose export directory"),ui->pathLineEdit->text(),QFileDialog::ShowDirsOnly);
    if(!path.isEmpty())
        ui->pathLineEdit->setText(path);
}


void ExportBatchDialog::accept()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    int lastExpt = s.value(QString("exptNum"),0).toInt();

    QList<int> range = Analysis::parseIntRanges(ui->rangeLineEdit->text(),lastExpt);
    if(range.isEmpty())
    {
        QMessageBox::critical(this,QString("Export error"),QString("Could not parse requested range. The maximum experiment number is %1.").arg(lastExpt));
        return;
    }

    std::sort(range.begin(),range.end());

    //test path
    QString fileName = ui->pathLineEdit->text() + QString("/expt%1.txt").arg(range.constFirst());
    QFile f(fileName);
    if(!f.open(QIODevice::WriteOnly))
    {
        QMessageBox::critical(this,QString("Export error"),QString("Could not write to %1. Please choose a different directory.").arg(ui->pathLineEdit->text()));
        return;
    }

    f.close();
    QList<int> errorList;

    QApplication::setOverrideCursor(Qt::BusyCursor);
    for(int i=0; i<range.size(); i++)
    {
        if(range.at(i) > lastExpt)
        {
            //this should never happen
            errorList.append(range.mid(i));
            break;
        }

        Experiment e(range.at(i));
        if(e.number()<1)
        {
            //could not load experiment
            errorList.append(range.at(i));
            continue;
        }

        fileName = ui->pathLineEdit->text() + QString("/expt%1.txt").arg(range.at(i));
        e.exportAscii(fileName);
        QApplication::processEvents();

    }
    QApplication::restoreOverrideCursor();

    if(!errorList.isEmpty())
    {
        QString errList = QString("%1").arg(errorList.constFirst());
        for(int i=1; i<errList.size(); i++)
            errList.append(QString(" %1").arg(errorList.at(i)));

        QMessageBox::warning(this,QString("Export warning"),QString("Could not export the following experiments: %1").arg(errList));
    }

    BlackChirp::setExportDir(ui->pathLineEdit->text());
    QDialog::accept();
}

void ExportBatchDialog::checkComplete()
{
    ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(!ui->rangeLineEdit->text().isEmpty());
}
