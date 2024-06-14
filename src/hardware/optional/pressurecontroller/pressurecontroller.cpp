#include <hardware/optional/pressurecontroller/pressurecontroller.h>

#include <QTimer>

using namespace BC::Key::PController;

PressureController::PressureController(const QString subKey, const QString name, CommunicationProtocol::CommType commType,
                                       bool ro, QObject *parent, bool threaded, bool critical) :
    HardwareObject(key,subKey,name,commType,parent,threaded,critical,d_count), d_readOnly(ro), d_config{subKey,d_count}
{
    set(readOnly,d_readOnly);

    d_count++;
}

PressureController::~PressureController()
{
}

double PressureController::readPressure()
{
    auto p = hwReadPressure();
    if(!isnan(p))
    {
        d_config.d_pressure = p;
        emit pressureUpdate(p,QPrivateSignal());
    }

    return p;
}

void PressureController::setPressureSetpoint(const double val)
{
    auto v = hwSetPressureSetpoint(val);
    if(!isnan(v))
        readPressureSetpoint();
}

void PressureController::readPressureSetpoint()
{
    auto v = hwReadPressureSetpoint();
    if(!isnan(v))
    {
        d_config.d_setPoint = v;
        emit pressureSetpointUpdate(v,QPrivateSignal());
    }
}

void PressureController::setPressureControlMode(bool enabled)
{
    hwSetPressureControlMode(enabled);
    readPressureControlMode();
}

void PressureController::readPressureControlMode()
{
    auto i = hwReadPressureControlMode();
    if(i<0)
        return;

    d_config.d_pressureControlMode = static_cast<bool>(i);
    emit pressureControlMode(static_cast<bool>(i),QPrivateSignal());
}

void PressureController::openGateValve()
{
    setPressureControlMode(false);
    hwOpenGateValve();
}

void PressureController::closeGateValve()
{
    setPressureControlMode(false);
    hwCloseGateValve();
}

PressureControllerConfig PressureController::getConfig() const
{
    //make a copy!
    PressureControllerConfig out(d_config);
    return out;
}

bool PressureController::prepareForExperiment(Experiment &e)
{
    auto wp = e.getOptHwConfig<PressureControllerConfig>(d_config.headerKey());
    if(auto p = wp.lock())
    {
        if(!qFuzzyCompare(d_config.d_setPoint,p->d_setPoint))
            setPressureSetpoint(p->d_setPoint);
        if(d_config.d_pressureControlMode != p->d_pressureControlMode)
            setPressureControlMode(p->d_pressureControlMode);
    }

    e.auxData()->registerKey(d_key,d_subKey,BC::Aux::PController::pressure);

    e.addOptHwConfig(d_config);

    return true;
}


AuxDataStorage::AuxDataMap PressureController::readAuxData()
{
    AuxDataStorage::AuxDataMap out;
    out.insert({BC::Aux::PController::pressure,readPressure()});
    return out;
}


void PressureController::initialize()
{
    p_readTimer = new QTimer(this);
    connect(p_readTimer,&QTimer::timeout,this,&PressureController::readPressure);
    connect(this,&PressureController::hardwareFailure,p_readTimer,&QTimer::stop);

    pcInitialize();
}

bool PressureController::testConnection()
{
    p_readTimer->stop();
    bool success = pcTestConnection();
    if(success)
        p_readTimer->start(get(readInterval,200));

    return success;
}


QStringList PressureController::validationKeys() const
{
    return {BC::Aux::PController::pressure};
}
