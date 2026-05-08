#include "pythongpibcontroller.h"

#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonGpibController, "Python GPIB Controller (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonGpibController, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Custom, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_SETTINGS(PythonGpibController)

// ============================================================================
// Constructor
// ============================================================================
PythonGpibController::PythonGpibController(const QString &label, QObject *parent) :
    GpibController(QString(PythonGpibController::staticMetaObject.className()), label, parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;
}

// ============================================================================
// initialize()
// ============================================================================
void PythonGpibController::initialize()
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
// testConnection()
// ============================================================================
bool PythonGpibController::testConnection()
{
    if (!testPythonConnection(p_comm)) {
        d_errorString = pythonErrorString();
        return false;
    }
    return true;
}

// ============================================================================
// readAddress()
// ============================================================================
bool PythonGpibController::readAddress()
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1] = "read_address"_L1;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains("error"_L1) &&
           resp["result"_L1].toBool(false);
}

// ============================================================================
// setAddress()
// ============================================================================
bool PythonGpibController::setAddress(int a)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1]  = "set_address"_L1;
    req["address"_L1] = a;
    auto resp = pu_process->sendRequest(req);
    if (!resp.contains("error"_L1) &&
        resp["result"_L1].toBool(false)) {
        d_currentAddress = a;
        return true;
    }
    return false;
}

// ============================================================================
// sleep()
// ============================================================================
void PythonGpibController::sleep(bool b)
{
    pythonSleep(b);
}

// ============================================================================
// gpibReadSettings()
// ============================================================================
void PythonGpibController::gpibReadSettings()
{
    pythonReadSettings();
}

