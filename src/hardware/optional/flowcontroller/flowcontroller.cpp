#include <hardware/optional/flowcontroller/flowcontroller.h>

using namespace BC::Key::Flow;

FlowController::FlowController(const QString subKey, const QString name, CommunicationProtocol::CommType commType,
                               QObject *parent, bool threaded, bool critical) :
    HardwareObject(flowController,subKey,name,commType,parent,threaded,critical),
    d_numChannels(getOrSetDefault(flowChannels,4))
{
    for(int i=0; i<d_numChannels; ++i)
        d_config.add({});

    setDefault(interval,333);
}

FlowController::~FlowController()
{

}

void FlowController::initialize()
{
    p_readTimer = new QTimer(this);
    connect(p_readTimer,&QTimer::timeout,this,&FlowController::poll);
    connect(this,&FlowController::hardwareFailure,p_readTimer,&QTimer::stop);

    fcInitialize();
}

bool FlowController::testConnection()
{
    p_readTimer->stop();
    p_readTimer->setInterval(get(interval,333));
    bool success = fcTestConnection();
    if(success)
    {
        p_readTimer->start();
        QTimer::singleShot(1000,this,&FlowController::readAll);
    }
    return success;
}

void FlowController::setChannelName(const int ch, const QString name)
{
    if(ch < d_config.size())
        d_config.set(ch,FlowConfig::Name,name);
}

void FlowController::setPressureControlMode(bool enabled)
{
    hwSetPressureControlMode(enabled);
    readPressureControlMode();
}

void FlowController::setFlowSetpoint(const int ch, const double val)
{
    if(ch < 0 || ch >= d_numChannels)
    {
        emit logMessage(QString("Invalid flow channel (%1) requested. Valid channels are 0-%2").arg(ch).arg(d_numChannels));
        return;
    }
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
    if(ch < 0 || ch >= d_numChannels)
    {
        emit logMessage(QString("Invalid flow channel (%1) requested. Valid channels are 0-%2").arg(ch).arg(d_numChannels));
        return;
    }
    double sp = hwReadFlowSetpoint(ch);
    if(sp > -1e-10)
    {
        d_config.set(ch,FlowConfig::Setpoint,sp);
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
    if(ch < 0 || ch >= d_numChannels)
    {
        emit logMessage(QString("Invalid flow channel (%1) requested. Valid channels are 0-%2").arg(ch).arg(d_numChannels));
        return;
    }
    double flow = hwReadFlow(ch);
    if(flow>-1.0)
    {
        d_config.set(ch,FlowConfig::Flow,flow);
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

void FlowController::poll()
{

}

void FlowController::readAll()
{
    for(int i=0; i<d_config.size(); i++)
    {
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
        if(d_config.setting(i,FlowConfig::Enabled).toBool())
            out.append(qMakePair(QString("flow.%1").arg(i),d_config.setting(i,FlowConfig::Flow)));
    }

    return out;
}
