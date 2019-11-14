#include "liflasercontroldoublespinbox.h"

#include <QApplication>
#include <QSettings>
#include <cmath>

LifLaserControlDoubleSpinBox::LifLaserControlDoubleSpinBox(QWidget *parent) : QDoubleSpinBox(parent)
{

}

void LifLaserControlDoubleSpinBox::configure(bool step)
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lifLaser"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());

    double min = s.value(QString("minPos"),200.0).toDouble();
    double max = s.value(QString("maxPos"),2000.0).toDouble();
    int decimals = s.value(QString("decimals"),2).toInt();
    QString units = s.value(QString("units"),QString("nm")).toString();

    s.endGroup();
    s.endGroup();

    setDecimals(decimals);
    if(step)
        setRange(pow(10.0,-decimals),max);
    else
        setRange(min,max);

    setSuffix(QString(" %1").arg(units));
}
