#include "pythontemperaturecontroller.h"

#include <QJsonObject>
#include <cmath>

#include <data/settings/hardwarekeys.h>
#include <hardware/core/hardwareregistration.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonTemperatureController, "Python Temperature Controller (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonTemperatureController, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_SETTINGS(PythonTemperatureController,
    {BC::Key::TC::numChannels, "Temperature Channels",
     "Number of temperature sensor input channels",
     4, 1, 16, HwSettingPriority::Required}
)

// ============================================================================
// Constructor / Destructor
// ============================================================================
PythonTemperatureController::PythonTemperatureController(const QString &label, QObject *parent) :
    TemperatureController(
        QString(PythonTemperatureController::staticMetaObject.className()),
        label,
        [&label]() -> uint {
            SettingsStorage s(BC::Key::hwKey(
                                  QString(TemperatureController::staticMetaObject.className()),
                                  label)
                              );
            auto n = s.get(BC::Key::TC::numChannels, 4u);
            return n;
        }(),
        parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;

    save();
}

// ============================================================================
// tcInitialize()
// ============================================================================
void PythonTemperatureController::tcInitialize()
{
    initPythonProcess(p_comm,
        [this](const QString &key, const QVariant &defaultVal) -> QVariant {
            return get(key, defaultVal);
        },
        [this](const QString &key, const QVariant &val) {
            set(key, val, true);
        }
    );

}

// ============================================================================
// tcTestConnection()
// ============================================================================
bool PythonTemperatureController::tcTestConnection()
{
    if (!testPythonConnection(p_comm)) {
        d_errorString = pythonErrorString();
        return false;
    }
    return true;
}

// ============================================================================
// readHwTemperature()
// ============================================================================
double PythonTemperatureController::readHwTemperature(const uint ch)
{
    if (!pu_process || !pu_process->isRunning())
        return std::nan("");

    QJsonObject req;
    req["method"_L1]  = "hw_read_temperature"_L1;
    req["channel"_L1] = static_cast<int>(ch);
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return std::nan("");

    return resp["result"_L1].toDouble(std::nan(""));
}

// ============================================================================
// sleep()
// ============================================================================
void PythonTemperatureController::sleep(bool b)
{
    pythonSleep(b);
}

