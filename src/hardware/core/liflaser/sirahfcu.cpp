#include <hardware/core/liflaser/sirahfcu.h>

#include <hardware/core/hardwareregistration.h>
#include <data/storage/enumcsvconvert.h>
#include <data/lif/lifunits.h>

#include <cmath>
#include <math.h>
#include <QThread>

#ifndef M_PI
#define M_PI 3.1415926535897323846
#endif

using namespace BC::Key::SirahFcu;
using namespace BC::Key::LifConvStage;
using namespace BC::LifConv;
using namespace BC::FcuCal;

// Register hardware implementation
REGISTER_HARDWARE_META(SirahFcu, "Sirah Frequency Conversion Unit")
REGISTER_HARDWARE_PROTOCOLS(SirahFcu, CommunicationProtocol::Rs232)
// Same binary protocol, no text terminator, as SirahCobra; the FCU has its
// own comm settings group via the normal comm-config dialog.
REGISTER_COMM_DEFAULTS(SirahFcu, CommunicationProtocol::Rs232,
    {BC::Key::Comm::timeout, 200},
    {BC::Key::Comm::termChar, QString("")})

// Override the base LifFreqConversionStage harmonic-order default for the
// common lone-doubler case. The registered op setting stays the inherited
// base default (NHG), matching conversionOp()'s pinned override below.
REGISTER_HARDWARE_SETTINGS(SirahFcu,
    {harmonic,   "Harmonic Order",       "Harmonic order N for this doubler; set once at profile creation",
     2,    1,          QVariant{}, HwSettingPriority::Required},
    {calScheme,  "Calibration Scheme",   "Tuning-curve model used to evaluate the doubling crystal's "
                                          "wavelength <-> motor position mapping",
     QVariant::fromValue(Scheme::Physical), QVariant{}, QVariant{}, HwSettingPriority::Important},
    {calCrystal, "Crystal Type",         "Doubling-crystal species; picks the Sellmeier equations for "
                                          "the Physical calibration scheme",
     QVariant::fromValue(CrystalType::BBO), QVariant{}, QVariant{}, HwSettingPriority::Important},
    {calInvert,  "Invert Phase Match",   "Select the alternate (-) branch of the Physical scheme's "
                                          "phase-match relation",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

// Sine-bar drive mechanics for the doubling-crystal motor stage. cutAngleDeg
// and temperature feed the Physical calibration scheme; the remaining
// entries are the sine-bar geometry and motor-control parameters shared by
// every scheme.
REGISTER_HARDWARE_ARRAY(SirahFcu, stages,
    "Crystal Stage Geometry", "Sine-bar tuning geometry for the doubling-crystal motor stage",
    HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(SirahFcu, stages,
    {{sStart,3000.0},
     {sHigh,12000.0},
     {sRamp,2400},
     {sMax,3300000},
     {sbls,24000},
     {sLeverLength,134.599318},
     {sLinearOffset,-76.543335},
     {sAngleOffset,31.329809},
     {sCutAngle,57.4},
     {sTemperature,293.0},
     {sPitch,-0.25},
     {sMotorResolution,4800},
    })

// Polynomial calibration scheme: imported forward (wavelength (nm) ->
// position) and inverse (position -> wavelength (nm)) coefficient lists, one
// row per order. Registered with no entries; populated by CSV import.
REGISTER_HARDWARE_ARRAY(SirahFcu, polyCoeffs,
    "Polynomial Coefficients", "Forward/inverse coefficient lists for the Polynomial calibration scheme",
    HwSettingPriority::Optional)

// Spline calibration scheme: imported (wavelength, position) point table.
// Registered with no entries; populated by CSV import.
REGISTER_HARDWARE_ARRAY(SirahFcu, splinePoints,
    "Spline Points", "Wavelength/position point table for the Spline calibration scheme",
    HwSettingPriority::Optional)

SirahFcu::SirahFcu(const QString& label, QObject *parent) :
    LifFreqConversionStage(QString(SirahFcu::staticMetaObject.className()), label, parent)
{
}

BC::LifConv::Op SirahFcu::conversionOp() const
{
    // A Sirah FCU is an N-th-harmonic-generation doubling crystal by device
    // identity, not a user-selectable operation.
    return Op::NHG;
}

void SirahFcu::initialize()
{
}

bool SirahFcu::testConnection()
{
    bool out = prompt();
    if(out)
    {
        //disable autoprompt
        p_comm->writeBinary(BC::Sirah::buildCommand(0x10));
        readPosition();
    }

    return out;
}

void SirahFcu::hwReadSettings()
{
    auto scheme = BC::CSV::enumFromVariant<Scheme>(
                get(calScheme, QVariant::fromValue(Scheme::Physical)), Scheme::Physical);

    switch(scheme)
    {
    case Scheme::Physical:
    {
        auto crystal = BC::CSV::enumFromVariant<CrystalType>(
                    get(calCrystal, QVariant::fromValue(CrystalType::BBO)), CrystalType::BBO);
        auto invert = get(calInvert, false);
        auto cutAngleDeg = getArrayValue(stages,0,sCutAngle,57.4);
        auto temperature = getArrayValue(stages,0,sTemperature,293.0);
        auto linearOffsetMm = getArrayValue(stages,0,sLinearOffset,-76.543335);
        auto angleOffsetDeg = getArrayValue(stages,0,sAngleOffset,31.329809);
        auto screwPitchMm = getArrayValue(stages,0,sPitch,-0.25);
        auto leverLengthMm = getArrayValue(stages,0,sLeverLength,134.599318);
        auto motorResolution = getArrayValue(stages,0,sMotorResolution,4800.0);

        d_calibration = FcuCalibration::physical(crystal, cutAngleDeg, temperature, linearOffsetMm,
                                                  angleOffsetDeg, screwPitchMm, leverLengthMm,
                                                  motorResolution, invert);
        break;
    }
    case Scheme::Polynomial:
    {
        std::vector<double> forwardCoeffs, inverseCoeffs;
        for(std::size_t i=0; i<getArraySize(polyCoeffs); i++)
        {
            auto order = static_cast<std::size_t>(getArrayValue(polyCoeffs,i,pcOrder,0));
            auto fwd = getArrayValue(polyCoeffs,i,pcForward,0.0);
            auto inv = getArrayValue(polyCoeffs,i,pcInverse,0.0);

            if(order >= forwardCoeffs.size())
                forwardCoeffs.resize(order+1, 0.0);
            if(order >= inverseCoeffs.size())
                inverseCoeffs.resize(order+1, 0.0);

            forwardCoeffs[order] = fwd;
            inverseCoeffs[order] = inv;
        }

        d_calibration = FcuCalibration::polynomial(forwardCoeffs, inverseCoeffs);
        break;
    }
    case Scheme::Spline:
    {
        std::vector<std::pair<double,double>> points;
        for(std::size_t i=0; i<getArraySize(splinePoints); i++)
        {
            auto wl = getArrayValue(splinePoints,i,spWavelength,0.0);
            auto pos = getArrayValue(splinePoints,i,spPosition,0.0);
            points.emplace_back(wl, pos);
        }

        d_calibration = FcuCalibration::spline(points);
        break;
    }
    }

    if(!d_calibration.isValid())
        hwWarn(d_calibration.errorString());
}

void SirahFcu::setPos(double localCm1)
{
    auto wl = fromCm1(localCm1, LaserUnit::Nm);

    if(!d_calibration.isValid())
    {
        hwError(u"Cannot set position to %1 cm-1 (%2 nm): %3"_s
                    .arg(localCm1,0,'f',3).arg(wl,0,'f',4).arg(d_calibration.errorString()));
        return;
    }

    if(!prompt())
    {
        hwError(u"Could not set position to %1 cm-1 (%2 nm)."_s
                    .arg(localCm1,0,'f',3).arg(wl,0,'f',4));
        return;
    }

    //calculate target position
    auto targetPos = static_cast<qint32>(round(d_calibration.wavelengthToPos(wl)));
    auto currentPos = d_status.m1Pos;
    auto delta = targetPos - currentPos;

    //if the calculated move is too small, just assume we're good enough
    if(qAbs(delta) < 10)
        return;

    //can we just move relative?
    //Conditions: need last move to be in same direction as backlash correction,
    //and distance should be less than backlash correction.
    auto backlash = getArrayValue(stages,0,sbls,-24000);
    if(d_status.lastMoveDir != 0 && (d_status.lastMoveDir*delta) > 0 && qAbs(delta) < qAbs(backlash))
    {
        moveRelative(delta);
        if(backlash > 0)
            d_status.lastMoveDir = 1;
        else
            d_status.lastMoveDir = -1;
    }
    else
    {
        if(moveAbsolute(targetPos - backlash))
        {
            moveRelative(backlash);
            if(backlash > 0)
                d_status.lastMoveDir = 1;
            else
                d_status.lastMoveDir = -1;
        }
        else
            d_status.lastMoveDir = 0;
    }
}

double SirahFcu::readPos()
{
    if(!prompt())
    {
        hwError("Could not read position."_L1);
        return -1.0;
    }

    auto wl = d_calibration.posToWavelength(d_status.m1Pos);
    if(!std::isfinite(wl))
    {
        hwError(u"Could not convert motor position %1 to a wavelength."_s.arg(d_status.m1Pos));
        return -1.0;
    }

    return toCm1(wl, LaserUnit::Nm);
}

bool SirahFcu::prompt()
{
    auto rp = BC::Sirah::buildCommand(0x17);
    p_comm->writeBinary(rp);
    auto resp = p_comm->readBytes(14,true);

    if(!BC::Sirah::parseStatus(resp, d_status))
    {
        d_errorString = QString("Received unexpected response (Hex: %1)").arg(QString(resp.toHex()));
        return false;
    }

    return true;
}

void SirahFcu::moveRelative(qint32 steps)
{
    quint8 dir = 0x01;
    if(steps < 0)
        dir = 0x02;

    auto s = qAbs(steps);
    QByteArray dat;
    dat.append(0x01);
    dat.append(dir);
    dat.append(static_cast<quint8>(s & 0x000000ff));
    dat.append(static_cast<quint8>((s & 0x0000ff00) >> 8));
    dat.append(static_cast<quint8>((s & 0x00ff0000) >> 16));
    dat.append(static_cast<quint8>((s & 0xff000000) >> 24));

    auto cmd = BC::Sirah::buildCommand(0x06,dat);

    p_comm->writeBinary(cmd);

    int waiting = 0;
    bool done = false;

    while(!done && waiting < 100)
    {
        thread()->msleep(50);

        if(!prompt())
            break;

        //bit 0 tells whether the motor is running
        if(d_status.m1Status % 2)
        {
            //motor is running; sleep thread and try again
            waiting++;
        }
        else
        {
            done = true;
            break;
        }
    }

    if(!done)
    {
        //stop motor
        p_comm->writeBinary(BC::Sirah::buildCommand(0x04));
        hwError("Did not set position successfully; stopped motor motion."_L1);
        emit hardwareFailure();
    }

}

bool SirahFcu::moveAbsolute(qint32 targetPos)
{
    QByteArray dat;
    dat.append(0x01);
    dat.append(static_cast<quint8>(targetPos & 0x000000ff));
    dat.append(static_cast<quint8>((targetPos & 0x0000ff00) >> 8));
    dat.append(static_cast<quint8>((targetPos & 0x00ff0000) >> 16));
    dat.append(static_cast<quint8>((targetPos & 0xff000000) >> 24));

    auto cmd = BC::Sirah::buildCommand(0x07,dat);

    p_comm->writeBinary(cmd);

    int waiting = 0;
    bool done = false;
    qint32 lastDiff = qAbs(targetPos - d_status.m1Pos);

    while(!done)
    {
        thread()->msleep(50);

        if(!prompt())
            break;

        if(waiting > 0)
        {
            auto d = qAbs(targetPos - d_status.m1Pos);
            if(d > lastDiff && d > 10)
            {
                hwDebug("Diff increased."_L1);
                break;
            }
            lastDiff = d;
        }

        //bit 0 tells whether the motor is running
        if(d_status.m1Status % 2)
        {
            //motor is running; sleep thread and try again
            waiting++;
        }
        else
        {
            done = true;
            break;
        }
    }

    if(!done)
    {
        //stop motor
        p_comm->writeBinary(BC::Sirah::buildCommand(0x04));
        hwError("Did not set position successfully; stopped motor motion."_L1);
        emit hardwareFailure();
    }

    return done;
}
