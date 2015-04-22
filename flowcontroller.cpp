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

    emit channelNameUpdate(ch,name);

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
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);

    s.beginReadArray(QString("channels"));
    for(int i=0; i<d_config.size(); i++)
    {
        s.setArrayIndex(i);
        d_config.set(i,FlowConfig::Name,s.value(QString("name"),QString("Ch%1").arg(i+1)));
        emit channelNameUpdate(i,d_config.setting(i,FlowConfig::Name).toString());

        readFlowSetpoint(i);
        readFlow(i);
    }
    s.endArray();
    s.endGroup();

    readPressureSetpoint();
    readPressure();
    readPressureControlMode();
}

