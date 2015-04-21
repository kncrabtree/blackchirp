#include "flowcontroller.h"

FlowController::FlowController(QObject *parent) : HardwareObject(parent), d_pressureControlMode(false)
{
    d_key = QString("flowController");
}

FlowController::~FlowController()
{

}

void FlowController::setChannelName(const int ch, const QString name)
{
    if(ch < d_config.size())
        d_config.set(ch,FlowConfig::Name,name);

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);

    s.beginWriteArray(QString("channels"));
    s.setArrayIndex(ch);
    s.setValue(QString("name"),name);
    s.endArray();
    s.endGroup();
}

void FlowController::readAll()
{
    for(int i=0; i<d_config.size(); i++)
    {
        readFlowSetpoint(i);
        readFlow(i);
    }

    readPressureSetpoint();
    readPressure();
    readPressureControlMode();
}

