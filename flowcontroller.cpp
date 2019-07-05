#include "flowcontroller.h"

FlowController::FlowController(QObject *parent) : HardwareObject(parent)
{
    d_key = QString("flowController");

}

FlowController::~FlowController()
{

}

void FlowController::initialize()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    s.beginReadArray(QString("channels"));
    for(int i=0;i<d_numChannels;i++)
    {
        s.setArrayIndex(i);
        d_config.add(0.0,s.value(QString("name"),QString("")).toString());
    }
    s.endArray();
    s.endGroup();
    s.endGroup();

    p_readTimer = new QTimer(this);
    connect(p_readTimer,&QTimer::timeout,this,&FlowController::poll);
    connect(this,&FlowController::hardwareFailure,p_readTimer,&QTimer::stop);
    updateInterval();

    fcInitialize();
}

bool FlowController::testConnection()
{
    p_readTimer->stop();
    updateInterval();
    bool success = fcTestConnection();
    if(success)
    {
        p_readTimer->start();
        readAll();
    }
    return success;
}

void FlowController::setChannelName(const int ch, const QString name)
{
    if(ch < d_config.size())
        d_config.set(ch,BlackChirp::FlowSettingName,name);

    emit channelNameUpdate(ch,name,QPrivateSignal());

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    s.beginWriteArray(QString("channels"));
    s.setArrayIndex(ch);
    s.setValue(QString("name"),name);
    s.endArray();
    s.endGroup();
    s.endGroup();
}

void FlowController::setPressureControlMode(bool enabled)
{
    hwSetPressureControlMode(enabled);
    readPressureControlMode();
}

void FlowController::setFlowSetpoint(const int ch, const double val)
{
    hwSetFlowSetpoint(ch,val);
    readFlowSetpoint(ch);
}

void FlowController::setPressureSetpoint(const double val)
{
    hwSetPressureSetpoint(val);
    readPressureSetpoint();
}

void FlowController::readFlowSetpoint(const int ch)
{
    double sp = hwReadFlowSetpoint(ch);
    if(sp > -1e-10)
    {
        d_config.set(ch,BlackChirp::FlowSettingSetpoint,sp);
        emit flowSetpointUpdate(ch,sp,QPrivateSignal());
    }
}

void FlowController::readPressureSetpoint()
{
    double sp = hwReadPressureSetpoint();
    if(sp > -1e-10)
    {
        d_config.setPressureSetpoint(sp);
        emit pressureSetpointUpdate(sp,QPrivateSignal());
    }

}

void FlowController::readFlow(const int ch)
{
    double flow = hwReadFlow(ch);
    if(flow>-1.0)
    {
        d_config.set(ch,BlackChirp::FlowSettingFlow,flow);
        emit flowUpdate(ch,flow,QPrivateSignal());
    }
}

void FlowController::readPressure()
{
    double pressure = hwReadPressure();
    if(pressure > -1.0)
    {
        d_config.setPressure(pressure);
        emit pressureUpdate(pressure,QPrivateSignal());
    }
}

void FlowController::readPressureControlMode()
{
    int ret = hwReadPressureControlMode();
    if(ret < 0)
        return;

    d_config.setPressureControlMode(ret==1);
    emit pressureControlMode(ret==1,QPrivateSignal());
}

void FlowController::updateInterval()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    int interval = s.value(QString("pollIntervalMs"),333).toInt();
    s.endGroup();
    s.endGroup();
    p_readTimer->setInterval(interval);
}

void FlowController::poll()
{

}

void FlowController::readAll()
{
    for(int i=0; i<d_config.size(); i++)
    {
        emit channelNameUpdate(i,d_config.setting(i,BlackChirp::FlowSettingName).toString(),QPrivateSignal());
        readFlow(i);
        readFlowSetpoint(i);
    }

    readPressureSetpoint();
    readPressure();
    readPressureControlMode();
}

QList<QPair<QString, QVariant> > FlowController::readAuxPlotData()
{
    QList<QPair<QString,QVariant>> out;
    out.append(qMakePair(QString("gasPressure"),d_config.pressure()));
    for(int i=0; i<d_config.size(); i++)
    {
        if(d_config.setting(i,BlackChirp::FlowSettingEnabled).toBool())
            out.append(qMakePair(QString("flow.%1").arg(i),d_config.setting(i,BlackChirp::FlowSettingFlow)));
    }

    return out;
}
