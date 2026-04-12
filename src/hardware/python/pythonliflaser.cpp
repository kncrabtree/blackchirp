#include "pythonliflaser.h"

#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>
#include <data/settings/hardwarekeys.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonLifLaser, "Python LIF Laser (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonLifLaser, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)

REGISTER_HARDWARE_SETTINGS(PythonLifLaser,
    {BC::Key::LifLaser::minPos, "Min Position", "Minimum laser wavelength/position", 250.0, QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::LifLaser::maxPos, "Max Position", "Maximum laser wavelength/position", 2000.0, QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::LifLaser::units, "Position Units", "Units for position display (e.g. nm, cm-1)", QString("nm"), QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::LifLaser::decimals, "Display Decimals", "Number of decimal places for position display", 2, 0, 8, HwSettingPriority::Optional},
    {BC::Key::LifLaser::hasFl, "Has Flashlamp", "Laser has a software-controlled flashlamp", true, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

// ============================================================================
// Constructor
// ============================================================================
PythonLifLaser::PythonLifLaser(const QString &label, QObject *parent) :
    LifLaser(QString(PythonLifLaser::staticMetaObject.className()), label, parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;

    save();
}

// ============================================================================
// initialize()
// ============================================================================
void PythonLifLaser::initialize()
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
            this, &PythonLifLaser::logMessage);
}

// ============================================================================
// testConnection()
// ============================================================================
bool PythonLifLaser::testConnection()
{
    if (!testPythonConnection(p_comm)) {
        d_errorString = pythonErrorString();
        return false;
    }

    readPosition();
    readFlashLamp();

    return true;
}

// ============================================================================
// readPos()
// ============================================================================
double PythonLifLaser::readPos()
{
    if (!pu_process || !pu_process->isRunning())
        return -1.0;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("read_pos");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return -1.0;

    return resp[QStringLiteral("result")].toDouble(-1.0);
}

// ============================================================================
// setPos()
// ============================================================================
void PythonLifLaser::setPos(double pos)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("set_pos");
    req[QStringLiteral("pos")]    = pos;
    pu_process->sendRequest(req);
}

// ============================================================================
// readFl()
// ============================================================================
bool PythonLifLaser::readFl()
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("read_fl");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return false;

    return resp[QStringLiteral("result")].toBool(false);
}

// ============================================================================
// setFl()
// ============================================================================
bool PythonLifLaser::setFl(bool en)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("set_fl");
    req[QStringLiteral("enabled")] = en;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return false;

    return resp[QStringLiteral("result")].toBool(false);
}

// ============================================================================
// readSettings()
// ============================================================================
void PythonLifLaser::readSettings()
{
    pythonReadSettings();
}

// ============================================================================
// sleep()
// ============================================================================
void PythonLifLaser::sleep(bool b)
{
    pythonSleep(b);
}

// ============================================================================
// forbiddenKeys()
// ============================================================================
QStringList PythonLifLaser::forbiddenKeys() const
{
    return LifLaser::forbiddenKeys() + pythonForbiddenKeys();
}
