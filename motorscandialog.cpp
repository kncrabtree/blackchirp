#include "motorscandialog.h"
#include "ui_motorscandialog.h"

#include <QSettings>
#include <QMessageBox>

MotorScanDialog::MotorScanDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::MotorScanDialog)
{
    ui->setupUi(this);

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("motorController"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    ui->xMinBox->setRange(s.value(QString("xMin"),-100.0).toDouble(),
                          s.value(QString("xMax"),100.0).toDouble());
    ui->xMaxBox->setRange(s.value(QString("xMin"),-100.0).toDouble(),
                          s.value(QString("xMax"),100.0).toDouble());
    ui->yMinBox->setRange(s.value(QString("yMin"),-100.0).toDouble(),
                          s.value(QString("yMax"),100.0).toDouble());
    ui->yMaxBox->setRange(s.value(QString("yMin"),-100.0).toDouble(),
                          s.value(QString("yMax"),100.0).toDouble());
    ui->zMinBox->setRange(s.value(QString("zMin"),-100.0).toDouble(),
                          s.value(QString("zMax"),100.0).toDouble());
    ui->zMaxBox->setRange(s.value(QString("zMin"),-100.0).toDouble(),
                          s.value(QString("zMax"),100.0).toDouble());
    s.endGroup();
    s.endGroup();

    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    connect(ui->xPointsBox,vc,this,&MotorScanDialog::validateBoxes);
    connect(ui->yPointsBox,vc,this,&MotorScanDialog::validateBoxes);
    connect(ui->zPointsBox,vc,this,&MotorScanDialog::validateBoxes);

    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    connect(ui->xMinBox,dvc,[=](double d){
        if(ui->xPointsBox->value() == 1)
            ui->xMaxBox->setValue(d);
    });
    connect(ui->yMinBox,dvc,[=](double d){
        if(ui->yPointsBox->value() == 1)
            ui->yMaxBox->setValue(d);
    });
    connect(ui->zMinBox,dvc,[=](double d){
        if(ui->zPointsBox->value() == 1)
            ui->zMaxBox->setValue(d);
    });


}

MotorScanDialog::~MotorScanDialog()
{
    delete ui;
}

void MotorScanDialog::setFromMotorScan(MotorScan ms)
{
    ui->xMinBox->setValue(ms.xVal(0));
    ui->xMaxBox->setValue(ms.xVal(ms.xPoints()-1));
    ui->xPointsBox->setValue(ms.xPoints());

    ui->yMinBox->setValue(ms.yVal(0));
    ui->yMaxBox->setValue(ms.yVal(ms.yPoints()-1));
    ui->yPointsBox->setValue(ms.yPoints());

    ui->zMinBox->setValue(ms.zVal(0));
    ui->zMaxBox->setValue(ms.zVal(ms.zPoints()-1));
    ui->zPointsBox->setValue(ms.zPoints());

    ui->shotsPerPointSpinBox->setValue(ms.shotsPerPoint());
}

MotorScan MotorScanDialog::toMotorScan()
{
    MotorScan out;

    out.setXPoints(ui->xPointsBox->value());
    out.setYPoints(ui->yPointsBox->value());
    out.setZPoints(ui->zPointsBox->value());

    double dx;
    if(ui->xPointsBox->value() > 1)
        dx = (ui->xMaxBox->value()-ui->xMinBox->value())/(static_cast<double>(ui->xPointsBox->value())-1.0);
    else
        dx = 0.0;

    double dy;
    if(ui->yPointsBox->value() > 1)
        dy = (ui->yMaxBox->value()-ui->yMinBox->value())/(static_cast<double>(ui->yPointsBox->value())-1.0);
    else
        dy = 0.0;

    double dz;
    if(ui->zPointsBox->value() > 1)
        dz = (ui->zMaxBox->value()-ui->zMinBox->value())/(static_cast<double>(ui->zPointsBox->value())-1.0);
    else
        dz = 0.0;

    out.setIntervals(ui->xMinBox->value(),ui->yMinBox->value(),ui->zMinBox->value(),dx,dy,dz);

    return out;

}

void MotorScanDialog::validateBoxes()
{
    if(ui->xPointsBox->value() == 1)
    {
        ui->xMaxBox->setValue(ui->xMinBox->value());
        ui->xMaxBox->setEnabled(false);
    }
    else
        ui->xMaxBox->setEnabled(true);

    if(ui->yPointsBox->value() == 1)
    {
        ui->yMaxBox->setValue(ui->yMinBox->value());
        ui->yMaxBox->setEnabled(false);
    }
    else
        ui->yMaxBox->setEnabled(true);

    if(ui->zPointsBox->value() == 1)
    {
        ui->zMaxBox->setValue(ui->zMinBox->value());
        ui->zMaxBox->setEnabled(false);
    }
    else
        ui->zMaxBox->setEnabled(true);
}


void MotorScanDialog::accept()
{
    if(ui->xPointsBox->value() > 1 && fabs(ui->xMinBox->value() - ui->xMaxBox->value()) < 0.1)
    {
        QMessageBox::critical(this,QString("Motor Scan Error"),QString("The beginning and ending X values must be different when the number of points is greater than 1."));
        return;
    }

    if(ui->yPointsBox->value() > 1 && fabs(ui->yMinBox->value() - ui->yMaxBox->value()) < 0.1)
    {
        QMessageBox::critical(this,QString("Motor Scan Error"),QString("The beginning and ending Y values must be different when the number of points is greater than 1."));
        return;
    }

    if(ui->zPointsBox->value() > 1 && fabs(ui->zMinBox->value() - ui->zMaxBox->value()) < 0.1)
    {
        QMessageBox::critical(this,QString("Motor Scan Error"),QString("The beginning and ending Z values must be different when the number of points is greater than 1."));
        return;
    }

    QDialog::accept();

}
