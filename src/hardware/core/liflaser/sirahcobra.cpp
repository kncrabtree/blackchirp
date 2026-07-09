#include "sirahcobra.h"
#include <hardware/core/hardwareregistration.h>
#include <data/lif/lifunits.h>

#include <math.h>
#include <QThread>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace BC::Key::LifLaser;
using namespace BC::LifConv;

// Register hardware implementation
REGISTER_HARDWARE_META(SirahCobra, "Sirah Cobra LIF Laser")
REGISTER_HARDWARE_PROTOCOLS(SirahCobra, CommunicationProtocol::Rs232)
// The Cobra speaks a binary protocol with no text terminator, so the
// termination character is deliberately empty.
REGISTER_COMM_DEFAULTS(SirahCobra, CommunicationProtocol::Rs232,
    {BC::Key::Comm::timeout, 200},
    {BC::Key::Comm::termChar, QString("")})

// minPos/maxPos are the grating fundamental (cm-1); the driver's native
// range is 450-700 nm, which is 1e7/700 = 14285.7 cm-1 (long-wavelength
// end) to 1e7/450 = 22222.2 cm-1 (short-wavelength end) — cm-1 grows as
// wavelength shrinks, so the nm bounds invert.
REGISTER_HARDWARE_SETTINGS(SirahCobra,
    {minPos,   "Min Position",     "Minimum grating fundamental position (cm-1; 700 nm)", 14285.7143, QVariant{}, QVariant{}, HwSettingPriority::Important, units},
    {maxPos,   "Max Position",     "Maximum grating fundamental position (cm-1; 450 nm)", 22222.2222, QVariant{}, QVariant{}, HwSettingPriority::Important, units},
    {decimals, "Display Decimals", "Number of decimal places for position display",       4,          0,          8,          HwSettingPriority::Optional},
    {hasFl,    "Has Flashlamp",    "Laser has a software-controlled flashlamp",           false,      QVariant{}, QVariant{}, HwSettingPriority::Optional}
)
REGISTER_HARDWARE_ARRAY(SirahCobra, stages,
    "Grating Stage Geometry", "Sine-bar tuning geometry for the grating motor stage",
    HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(SirahCobra, stages,
    {{sStart,3000.0},
     {sHigh,12000.0},
     {sRamp,2400},
     {sMax,3300000},
     {sbls,24000},
     {sLeverLength,134.599318},
     {sLinearOffset,-76.543335},
     {sAngleOffset,31.329809},
     {sGrazingAngle,85.0},
     {sGrooves,2414.0},
     {sPitch,-0.25},
     {sMotorResolution,4800},
    })

SirahCobra::SirahCobra(const QString& label, QObject *parent)
    : LifLaser(QString(SirahCobra::staticMetaObject.className()), label, parent)
{
}

void SirahCobra::initialize()
{
}

bool SirahCobra::testConnection()
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

double SirahCobra::readPos()
{
    if(!prompt())
    {
        hwError("Could not read position."_L1);
        return -1.0;
    }

    auto wl = posToWavelength(d_status.m1Pos);
    return toCm1(wl, LaserUnit::Nm);
}

void SirahCobra::setPos(double pos)
{
    auto wl = fromCm1(pos, LaserUnit::Nm);

    if(!prompt())
    {
        hwError(u"Could not set position to %1 cm-1 (%2 nm)."_s
                    .arg(pos,0,'f',3).arg(wl,0,'f',get(decimals,2)));
        return;
    }

    //calculate target position
    auto targetPos = wavelengthToPos(wl);
    auto currentPos = d_status.m1Pos;
    auto delta = targetPos - currentPos;

    //if the calculated move is too small, just assume we're good enough
    if(qAbs(delta) < 10)
        return;

    //can we just move relative?
    //Conditions: need last move to be in same direction as backlash correction,
    //and distance should be less than backlash correction.
    auto backlash = getArrayValue(stages,0,sbls,24000);
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

bool SirahCobra::readFl()
{
    return true;
}

bool SirahCobra::setFl(bool en)
{
    Q_UNUSED(en)
    return true;
}


void SirahCobra::lifLaserReadSettings()
{
    d_params.clear();
    for(uint i=0; i<getArraySize(stages); i++)
    {
        TuningParameters tp;
        tp.angOff = getArrayValue(stages,i,sAngleOffset).toDouble()/180*M_PI;
        tp.grazAng = getArrayValue(stages,i,sGrazingAngle).toDouble()/180*M_PI;
        tp.grooves = getArrayValue(stages,i,sGrooves).toDouble();
        tp.lLen = getArrayValue(stages,i,sLeverLength).toDouble();
        tp.linOff = getArrayValue(stages,i,sLinearOffset).toDouble();
        tp.mRes = getArrayValue(stages,i,sMotorResolution).toDouble();
        tp.pitch = getArrayValue(stages,i,sPitch).toDouble();
        d_params.push_back(tp);
    }
}

bool SirahCobra::prompt()
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

double SirahCobra::posToWavelength(qint32 pos, uint stage)
{
    if(stage >= d_params.size())
        return 0.0;

    const auto &tp = d_params.at(stage);
    auto x = tp.linOff - (tp.pitch/tp.mRes)*static_cast<double>(pos);
    auto phi_o = tp.angOff - asin(x/tp.lLen);
    auto wl = (sin(tp.grazAng) + sin(phi_o))/tp.grooves;

    return wl *1e6;
}

qint32 SirahCobra::wavelengthToPos(double wl, uint stage)
{
    if(stage >= d_params.size())
        return 0;

    const auto &tp = d_params.at(stage);
    auto phi_o = asin(tp.grooves*wl/1e6 - sin(tp.grazAng));
    auto x = tp.linOff - tp.lLen*sin(tp.angOff - phi_o);
    auto p = tp.mRes/tp.pitch*x;

    return static_cast<qint32>(round(p));
}

void SirahCobra::moveRelative(qint32 steps)
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

        // hwDebug(u"Target (rel): %1, Current: %2"_s.arg(steps).arg(d_status.m1Pos));

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

bool SirahCobra::moveAbsolute(qint32 targetPos)
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
        // hwDebug(u"Target: %1, Current: %2"_s.arg(targetPos).arg(d_status.m1Pos));

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
