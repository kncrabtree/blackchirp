#include <hardware/optional/pressurecontroller/pressurecontroller.h>

#include <QTimer>
#include <data/settings/hardwarekeys.h>
#include <hardware/core/hardwareregistration.h>

using namespace BC::Key::PController;

REGISTER_HARDWARE_BASE(PressureController,
    {min,          "Min Pressure",       "Minimum pressure reading (display range lower bound).",    -1.0,           QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {max,          "Max Pressure",       "Maximum pressure reading (display range upper bound).",    20.0,           0.0,        QVariant{}, HwSettingPriority::Optional},
    {decimals,     "Display Decimals",   "Number of decimal places in pressure display.",            4,              0,          8,          HwSettingPriority::Optional},
    {units,        "Pressure Units",     "Pressure units for display.",                              QString("Torr"), QVariant{}, QVariant{}, HwSettingPriority::Important},
    {readInterval, "Read Interval (ms)", "Polling interval for pressure readings in milliseconds.", 200,            1,          QVariant{}, HwSettingPriority::Optional},
    {hasValve,     "Has Valve",          "Device includes a controlled valve output.",               true,           QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

PressureController::PressureController(const QString& impl, const QString& label, bool ro, QObject *parent) :
    HardwareObject(QString(PressureController::staticMetaObject.className()), impl, label, parent), d_readOnly(ro), d_config{BC::Key::hwKey(QString(PressureController::staticMetaObject.className()), label)}
{
    set(readOnly,d_readOnly);
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

    e.auxData()->registerKey(d_key,BC::Aux::PController::pressure);

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

void PressureController::prepareForShutdown()
{
    if(p_readTimer)
        p_readTimer->stop();
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
