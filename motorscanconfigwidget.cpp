#include "motorscanconfigwidget.h"
#include "ui_motorscanconfigwidget.h"

#include <QMessageBox>
#include <math.h>

MotorScanConfigWidget::MotorScanConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MotorScanConfigWidget)
{
    ui->setupUi(this);


    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("motorController"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    s.beginReadArray(QString("channels"));
    s.setArrayIndex(0);
    ui->xMinBox->setRange(s.value(QString("min"),-100.0).toDouble(),
                          s.value(QString("max"),100.0).toDouble());
    ui->xMaxBox->setRange(s.value(QString("min"),-100.0).toDouble(),
                          s.value(QString("max"),100.0).toDouble());
    s.setArrayIndex(1);
    ui->yMinBox->setRange(s.value(QString("min"),-100.0).toDouble(),
                          s.value(QString("max"),100.0).toDouble());
    ui->yMaxBox->setRange(s.value(QString("min"),-100.0).toDouble(),
                          s.value(QString("max"),100.0).toDouble());
    s.setArrayIndex(2);
    ui->zMinBox->setRange(s.value(QString("min"),-100.0).toDouble(),
                          s.value(QString("max"),100.0).toDouble());
    ui->zMaxBox->setRange(s.value(QString("min"),-100.0).toDouble(),
                          s.value(QString("max"),100.0).toDouble());
    s.endArray();
    s.endGroup();
    s.endGroup();

    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    connect(ui->xPointsBox,vc,this,&MotorScanConfigWidget::validateBoxes);
    connect(ui->yPointsBox,vc,this,&MotorScanConfigWidget::validateBoxes);
    connect(ui->zPointsBox,vc,this,&MotorScanConfigWidget::validateBoxes);

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

MotorScanConfigWidget::~MotorScanConfigWidget()
{
    delete ui;
}

void MotorScanConfigWidget::setFromMotorScan(MotorScan ms)
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

    ui->scopeConfigWidget->setFromConfig(ms.scopeConfig());
}

MotorScan MotorScanConfigWidget::toMotorScan()
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
    out.setShotsPerPoint(ui->shotsPerPointSpinBox->value());

    out.setScopeConfig(ui->scopeConfigWidget->toConfig());

    return out;
}

void MotorScanConfigWidget::validateBoxes()
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

bool MotorScanConfigWidget::validatePage()
{
    if(ui->xPointsBox->value() > 1 && fabs(ui->xMinBox->value() - ui->xMaxBox->value()) < 0.1)
    {
        QMessageBox::critical(this,QString("Motor Scan Error"),QString("The beginning and ending X values must be different when the number of points is greater than 1."));
        return false;
    }

    if(ui->yPointsBox->value() > 1 && fabs(ui->yMinBox->value() - ui->yMaxBox->value()) < 0.1)
    {
        QMessageBox::critical(this,QString("Motor Scan Error"),QString("The beginning and ending Y values must be different when the number of points is greater than 1."));
        return false;
    }

    if(ui->zPointsBox->value() > 1 && fabs(ui->zMinBox->value() - ui->zMaxBox->value()) < 0.1)
    {
        QMessageBox::critical(this,QString("Motor Scan Error"),QString("The beginning and ending Z values must be different when the number of points is greater than 1."));
        return false;
    }

    return true;
}
