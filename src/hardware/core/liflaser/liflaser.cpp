#include <hardware/core/liflaser/liflaser.h>

#include <hardware/core/hardwareregistration.h>

using namespace BC::Key::LifLaser;

REGISTER_HARDWARE_BASE(LifLaser,
    {minPos,   "Min Position",     "Minimum laser wavelength/position",             250.0,      QVariant{}, QVariant{}, HwSettingPriority::Important},
    {maxPos,   "Max Position",     "Maximum laser wavelength/position",             2000.0,     QVariant{}, QVariant{}, HwSettingPriority::Important},
    {units,    "Position Units",   "Units for position display (e.g. nm, cm-1)",   QString("nm"), QVariant{}, QVariant{}, HwSettingPriority::Important},
    {decimals, "Display Decimals", "Number of decimal places for position display", 2,          0,          8,          HwSettingPriority::Optional},
    {hasFl,    "Has Flashlamp",    "Laser has a software-controlled flashlamp",     true,       QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

LifLaser::LifLaser(const QString& impl, const QString& label, QObject *parent) :
    HardwareObject(QString(LifLaser::staticMetaObject.className()), impl, label, parent)
{
    d_threaded = true;
}

LifLaser::~LifLaser()
{

}

double LifLaser::readPosition()
{
    double out = readPos();
    emit laserPosUpdate(out);
    if(out < 0.0)
    {
        hwError("Could not read position."_L1);
        emit hardwareFailure();
    }

    return out;
}

double LifLaser::setPosition(const double pos)
{
    using namespace BC::Key::LifLaser;
    auto minp = get(minPos,200.0);
    auto maxp = get(maxPos,2000.0);
    if(pos < minp || pos > maxp)
    {
        auto d = get(decimals,2);
        hwError(u"Requested position (%1 %2) is outside the allowed range of %3 %2 - %4 %2."_s.arg(pos,0,'f',d).arg(get(units, "nm").toString()).arg(minp,0,'f',d).arg(maxp,0,'f',d));
        emit hardwareFailure();
        return -1.0;
    }

    setPos(pos);

    return readPosition();
}

bool LifLaser::readFlashLamp()
{
    auto out = readFl();
    emit laserFlashlampUpdate(out);
    return out;
}

bool LifLaser::setFlashLamp(bool en)
{
    if(setFl(en))
    {
        readFlashLamp();
        return true;
    }

    return false;
}


void LifLaser::hwReadSettings()
{
    lifLaserReadSettings();
}

bool LifLaser::hwPrepareForExperiment(Experiment &exp)
{
    if(exp.lifEnabled())
        d_autoDisable = exp.lifConfig()->d_disableFlashlamp;

    return true;
}

void LifLaser::beginAcquisition()
{
    setFlashLamp(true);
}

void LifLaser::endAcquisition()
{
    if(d_autoDisable)
        setFlashLamp(false);
}
