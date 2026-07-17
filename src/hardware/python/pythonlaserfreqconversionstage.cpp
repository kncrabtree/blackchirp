#include "pythonlaserfreqconversionstage.h"

#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>
#include <data/settings/hardwarekeys.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonLaserFreqConversionStage, "Python Laser Frequency Conversion Stage (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonLaserFreqConversionStage, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Gpib, CommunicationProtocol::Custom, CommunicationProtocol::Virtual)

// ============================================================================
// Constructor
// ============================================================================
PythonLaserFreqConversionStage::PythonLaserFreqConversionStage(const QString &label, QObject *parent) :
    LaserFreqConversionStage(QString(PythonLaserFreqConversionStage::staticMetaObject.className()), label, parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;

    save();
}

// ============================================================================
// initialize()
// ============================================================================
void PythonLaserFreqConversionStage::initialize()
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
bool PythonLaserFreqConversionStage::testConnection()
{
    if (!testPythonConnection(p_comm)) {
        d_errorString = pythonErrorString();
        return false;
    }

    readPosition();

    return true;
}

// ============================================================================
// readPos()
// ============================================================================
double PythonLaserFreqConversionStage::readPos()
{
    if (!pu_process || !pu_process->isRunning())
        return -1.0;

    QJsonObject req;
    req["method"_L1] = "read_pos"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return -1.0;

    return resp["result"_L1].toDouble(-1.0);
}

// ============================================================================
// setPos()
// ============================================================================
void PythonLaserFreqConversionStage::setPos(double localCm1)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req["method"_L1] = "set_pos"_L1;
    req["pos"_L1]    = localCm1;
    pu_process->sendRequest(req);
}

// ============================================================================
// hwReadSettings()
// ============================================================================
void PythonLaserFreqConversionStage::hwReadSettings()
{
    pythonReadSettings();
}

// ============================================================================
// sleep()
// ============================================================================
void PythonLaserFreqConversionStage::sleep(bool b)
{
    pythonSleep(b);
}

