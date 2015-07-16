#include "ioboardconfigdialog.h"
#include "ui_ioboardconfigdialog.h"
#include <QSettings>
#include <QMessageBox>

IOBoardConfigDialog::IOBoardConfigDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::IOBoardConfigDialog), d_key(QString("ioboard"))
{
    ui->setupUi(this);

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());

    ui->serialNoSpinBox->setValue(s.value(QString("serialNo"),3).toInt());

    s.endGroup();
    s.endGroup();

    auto intVc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);

    connect(ui->serialNoSpinBox,intVc,this,&IOBoardConfigDialog::dirtySerialNumber);
    connect(ui->testConnectionButton,&QPushButton::clicked,this,&IOBoardConfigDialog::testConnectionCallback);

}

IOBoardConfigDialog::~IOBoardConfigDialog()
{
    delete ui;
}

void IOBoardConfigDialog::dirtySerialNumber()
{
    ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
}

void IOBoardConfigDialog::testConnectionCallback()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    s.setValue(QString("serialNo"),ui->serialNoSpinBox->value());
    s.endGroup();
    s.endGroup();
    s.sync();

    setCursor(Qt::BusyCursor);
    setEnabled(false);
    emit testConnection(QString(""),d_key);
}

void IOBoardConfigDialog::testComplete(QString device, bool success, QString msg)
{
    //configure ui
    setEnabled(true);
    setCursor(QCursor());
    ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);

    if(success)
        QMessageBox::information(this,QString("Connection Successful"),
                            QString("%1 connected successfully!").arg(device),QMessageBox::Ok);
    else
        QMessageBox::critical(this,QString("Connection failed"),
                              QString("%1 connection failed!\n%2").arg(device).arg(msg),QMessageBox::Ok);
}

void IOBoardConfigDialog::accept()
{
//    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
//    s.beginGroup(d_key);
//    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
//    s.endGroup();
//    s.endGroup();
//    s.sync();

    QDialog::accept();
}
