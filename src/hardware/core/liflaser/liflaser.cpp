#include <hardware/core/liflaser/liflaser.h>

#include <hardware/core/hardwareregistration.h>
#include <data/storage/enumcsvconvert.h>

using namespace BC::Key::LifLaser;
using namespace BC::LifConv;

// minPos/maxPos are the grating fundamental's native range (cm⁻¹); the old
// 250-2000 nm span becomes 5000-40000 cm⁻¹ (cm⁻¹ avoids the reciprocal
// min/max flip nm carries). Dye-laser users think in nm, so the display
// unit defaults to Nm even though the internal value is always cm⁻¹.
REGISTER_HARDWARE_BASE(LifLaser,
    {minPos,   "Min Position",     "Minimum laser fundamental position (cm-1)",     5000.0,     QVariant{}, QVariant{}, HwSettingPriority::Important},
    {maxPos,   "Max Position",     "Maximum laser fundamental position (cm-1)",     40000.0,    QVariant{}, QVariant{}, HwSettingPriority::Important},
    {units,    "Position Units",   "Units for position display (e.g. nm, cm-1)",   QVariant::fromValue(LaserUnit::Nm), QVariant{}, QVariant{}, HwSettingPriority::Important},
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
    double fundamental = readPos();
    if(fundamental < 0.0)
    {
        hwError("Could not read position."_L1);
        emit hardwareFailure();
        return -1.0;
    }

    double out = d_conversion.laserToOutput(fundamental);
    emit laserPosUpdate(out);
    return out;
}

double LifLaser::setPosition(const double pos)
{
    double fundamental = d_conversion.outputToLaser(pos);
    auto minp = get(minPos,5000.0);
    auto maxp = get(maxPos,40000.0);
    if(fundamental < minp || fundamental > maxp)
    {
        auto d = get(decimals,2);
        auto u = displayUnit();
        hwError(u"Requested position (%1 %2) is outside the allowed range of %3 %2 - %4 %2."_s
                    .arg(fromCm1(fundamental,u),0,'f',d)
                    .arg(unitLabel(u))
                    .arg(fromCm1(minp,u),0,'f',d)
                    .arg(fromCm1(maxp,u),0,'f',d));
        emit hardwareFailure();
        return -1.0;
    }

    setPos(fundamental);

    return readPosition();
}

void LifLaser::setConversion(const LifConversion &c)
{
    d_conversion = c;
}

BC::LifConv::LaserUnit LifLaser::displayUnit() const
{
    return BC::CSV::enumFromVariant<LaserUnit>(get(units, QVariant::fromValue(LaserUnit::Nm)), LaserUnit::Nm);
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
