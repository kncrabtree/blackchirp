#include <hardware/optional/flowcontroller/flowcontroller.h>
#include <hardware/core/hardwareregistration.h>

using namespace BC::Key;

REGISTER_HARDWARE_SETTINGS(FlowController,
    {BC::Key::Flow::interval, "Poll Interval (ms)",
     "Interval between flow controller readback queries in milliseconds.",
     333, 1, QVariant{}, HwSettingPriority::Optional}
)

FlowController::FlowController(const QString& impl, const QString& label, QObject *parent) :
    HardwareObject(QString(FlowController::staticMetaObject.className()), impl, label, parent),
    d_config(BC::Key::hwKey(QString(FlowController::staticMetaObject.className()), label)),
    d_numChannels(get(Flow::flowChannels,4))
{
    for(int i=0; i<d_numChannels; ++i)
        d_config.addCh({});

    if(containsArray(Flow::channels))
    {
        for(int i=0; i<d_numChannels; i++)
            d_config.setCh(i,FlowConfig::Name,getArrayValue(Flow::channels,i,Flow::chName,QString("Ch%1").arg(i+1)));
    }
}

FlowController::~FlowController()
{
    setArray(Flow::channels, {});

    for(int i=0; i<d_numChannels; i++)
    {
        auto n = d_config.setting(i,FlowConfig::Name).toString();
        if(n.isEmpty())
            n = QString("Ch%1").arg(i+1);
        SettingsMap m {
            {Flow::chName,n},
        };
        appendArrayMap(Flow::channels,m);
    }
    save();
}

void FlowController::readSettings()
{
    using namespace BC::Key::Flow;
    int newCount = get(flowChannels, d_numChannels);
    if (newCount == d_numChannels)
        return;

    // Rebuild d_config preserving data for channels that still exist
    FlowConfig newConfig(d_config.headerKey());
    int preserve = qMin(d_numChannels, newCount);
    for (int i = 0; i < newCount; ++i)
    {
        if (i < preserve)
            newConfig.addCh(d_config.setting(i, FlowConfig::Setpoint).toDouble(),
                            d_config.setting(i, FlowConfig::Name).toString());
        else
            newConfig.addCh(0.0, getArrayValue(channels, i, chName, QString("Ch%1").arg(i+1)));
    }
    d_config = newConfig;
    d_numChannels = newCount;

    if (d_nextRead >= d_numChannels)
        d_nextRead = -1;
}

void FlowController::setAll(const FlowConfig &c)
{
    for(int i=0; i<c.size(); ++i)
    {
        setChannelName(i,c.setting(i,FlowConfig::Name).toString());
        setFlowSetpoint(i,c.setting(i,FlowConfig::Setpoint).toDouble());
    }
    setPressureSetpoint(c.d_pressureSetpoint);
    setPressureControlMode(c.d_pressureControlMode);
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
    p_readTimer->setInterval(get(Flow::interval,333));
    bool success = fcTestConnection();
    if(success)
    {
        p_readTimer->start();
        QTimer::singleShot(1000,this,&FlowController::readAll);
    }
    return success;
}

bool FlowController::prepareForExperiment(Experiment &e)
{
    auto wp = e.getOptHwConfig<FlowConfig>(d_config.headerKey());
    if(auto p = wp.lock())
        setAll(*p);

    if(!isConnected())
        return false;

    e.auxData()->registerKey(d_key,BC::Aux::Flow::pressure);
    for(int i=0; i<d_numChannels; i++)
    {
        if(d_config.setting(i,FlowConfig::Enabled).toBool())
        {
            auto n = d_config.setting(i,FlowConfig::Name).toString();
            if(n.isEmpty())
                e.auxData()->registerKey(d_key,BC::Aux::Flow::flow.arg(i+1));
            else
                e.auxData()->registerKey(d_key,n+"."+BC::Aux::Flow::flow.arg(i+1));
        }
    }

    e.addOptHwConfig(d_config);

    return true;
}

void FlowController::setChannelName(const int ch, const QString name)
{
    if(ch < d_numChannels)
        d_config.setCh(ch,FlowConfig::Name,name);
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
        emit logMessage(QString("Invalid flow channel (%1) requested. Valid channels are 0-%2").arg(ch).arg(d_numChannels-1));
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
        d_config.setCh(ch,FlowConfig::Setpoint,sp);
        emit flowSetpointUpdate(ch,sp,QPrivateSignal());
    }
}

void FlowController::readPressureSetpoint()
{
    double sp = hwReadPressureSetpoint();
    if(sp > -1e-10)
    {
        d_config.d_pressureSetpoint = sp;
        emit pressureSetpointUpdate(sp,QPrivateSignal());
    }

}

void FlowController::readFlow(const int ch)
{
    if(ch < 0 || ch >= d_numChannels)
    {
        emit logMessage(QString("Invalid flow channel (%1) requested. Valid channels are 0-%2").arg(ch).arg(d_numChannels-1));
        return;
    }
    double flow = hwReadFlow(ch);
    if(flow>-1.0)
    {
        d_config.setCh(ch,FlowConfig::Flow,flow);
        emit flowUpdate(ch,flow,QPrivateSignal());
    }
}

void FlowController::readPressure()
{
    double pressure = hwReadPressure();
    if(pressure > -1.0)
    {
        d_config.d_pressure = pressure;
        emit pressureUpdate(pressure,QPrivateSignal());
    }
}

void FlowController::readPressureControlMode()
{
    int ret = hwReadPressureControlMode();
    if(ret < 0)
        return;

    d_config.d_pressureControlMode = (ret==1);
    emit pressureControlMode(d_config.d_pressureControlMode,QPrivateSignal());
}

void FlowController::poll()
{
    if (d_nextRead < 0 || d_nextRead >= d_numChannels) {
        readPressure();
        d_nextRead = 0;
    } else {
        readFlow(d_nextRead);
        d_nextRead++;
    }
}

void FlowController::readAll()
{
    for(int i=0; i<d_numChannels; i++)
    {
        readFlow(i);
        readFlowSetpoint(i);
    }

    readPressureSetpoint();
    readPressure();
    readPressureControlMode();
}

AuxDataStorage::AuxDataMap FlowController::readAuxData()
{
    AuxDataStorage::AuxDataMap out;
    out.insert({BC::Aux::Flow::pressure,d_config.d_pressure});
    for(int i=0; i<d_numChannels; ++i)
    {
        auto n = d_config.setting(i,FlowConfig::Name).toString();
        if(d_config.setting(i,FlowConfig::Enabled).toBool())
        {
            if(n.isEmpty())
                out.insert({BC::Aux::Flow::flow.arg(i+1),d_config.setting(i,FlowConfig::Flow)});
            else
                out.insert({n+"."+BC::Aux::Flow::flow.arg(i+1),d_config.setting(i,FlowConfig::Flow)});
        }
    }

    return out;
}


QStringList FlowController::validationKeys() const
{
    QStringList out;
    for(int i=0; i<d_numChannels; ++i)
        out.append(BC::Aux::Flow::flow.arg(i));

    return out;
}


QStringList FlowController::forbiddenKeys() const
{
    return {BC::Key::Flow::flowChannels};
}
