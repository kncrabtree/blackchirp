#include "pythongpibcontroller.h"

#ifdef BC_PYTHON_HARDWARE

#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonGpibController, "Python GPIB Controller (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonGpibController, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_PARAMS(PythonGpibController)

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
// configParams()
// ============================================================================
QVector<HwConfigParam> PythonGpibController::configParams()
{
    return {};
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

    connect(pu_process.get(), &PythonProcess::logMessage,
            this, &PythonGpibController::logMessage);
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
    req[QStringLiteral("method")] = QStringLiteral("read_address");
    auto resp = pu_process->sendRequest(req);
    return !resp.contains(QStringLiteral("error")) &&
           resp[QStringLiteral("result")].toBool(false);
}

// ============================================================================
// setAddress()
// ============================================================================
bool PythonGpibController::setAddress(int a)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("set_address");
    req[QStringLiteral("address")] = a;
    auto resp = pu_process->sendRequest(req);
    if (!resp.contains(QStringLiteral("error")) &&
        resp[QStringLiteral("result")].toBool(false)) {
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
// readSettings()
// ============================================================================
void PythonGpibController::readSettings()
{
    pythonReadSettings();
}

// ============================================================================
// forbiddenKeys()
// ============================================================================
QStringList PythonGpibController::forbiddenKeys() const
{
    return GpibController::forbiddenKeys() + pythonForbiddenKeys();
}

#endif // BC_PYTHON_HARDWARE
