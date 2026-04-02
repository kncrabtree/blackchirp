#include "pythontemperaturecontroller.h"

#ifdef BC_PYTHON_HARDWARE

#include <QJsonObject>
#include <cmath>

#include <data/settings/hardwarekeys.h>
#include <hardware/core/hardwareregistration.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonTemperatureController, "Python Temperature Controller (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonTemperatureController, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_PARAMS(PythonTemperatureController)

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
// configParams()
// ============================================================================
QVector<HwConfigParam> PythonTemperatureController::configParams()
{
    return {
        { BC::Key::TC::numChannels, QStringLiteral("Temperature Channels"), QVariant(4), QVariant(1), QVariant(16) },
    };
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

    connect(pu_process.get(), &PythonProcess::logMessage,
            this, &PythonTemperatureController::logMessage);
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
    req[QStringLiteral("method")]  = QStringLiteral("hw_read_temperature");
    req[QStringLiteral("channel")] = static_cast<int>(ch);
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return std::nan("");

    return resp[QStringLiteral("result")].toDouble(std::nan(""));
}

// ============================================================================
// sleep()
// ============================================================================
void PythonTemperatureController::sleep(bool b)
{
    pythonSleep(b);
}

// ============================================================================
// forbiddenKeys()
// ============================================================================
QStringList PythonTemperatureController::forbiddenKeys() const
{
    return TemperatureController::forbiddenKeys() + pythonForbiddenKeys();
}

#endif // BC_PYTHON_HARDWARE
