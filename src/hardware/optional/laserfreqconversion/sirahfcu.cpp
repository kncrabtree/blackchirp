#include <hardware/optional/laserfreqconversion/sirahfcu.h>

#include <hardware/core/hardwareregistration.h>
#include <data/storage/enumcsvconvert.h>
#include <data/lif/lifunits.h>

#include <cmath>
#include <math.h>
#include <QElapsedTimer>
#include <QThread>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace BC::Key::SirahFcu;
using namespace BC::Key::LaserConvStage;
using namespace BC::LifConv;
using namespace BC::FcuCal;

// Register hardware implementation
REGISTER_HARDWARE_META(SirahFcu, "Sirah Frequency Conversion Unit")
REGISTER_HARDWARE_PROTOCOLS(SirahFcu, CommunicationProtocol::Rs232)
// Binary protocol (BC::Autotracker, see autotrackerprotocol.h) with no text
// terminator; the FCU has its own comm settings group via the normal
// comm-config dialog. The 200 ms default timeout suits the short
// query/response commands (Get Position, Identify, Error); a Goto Position
// move with Wait=0 (see moveAbsolute()/moveRelative()) reads its
// acknowledgment with a separately extended timeout instead of relying on
// this default.
REGISTER_COMM_DEFAULTS(SirahFcu, CommunicationProtocol::Rs232,
    {BC::Key::Comm::timeout, 200},
    {BC::Key::Comm::termChar, QString("")})

// Override the base LaserFreqConversionStage harmonic-order default for the
// common lone-doubler case. The registered op setting stays the inherited
// base default (NHG), matching conversionOp()'s pinned override below.
REGISTER_HARDWARE_SETTINGS(SirahFcu,
    {harmonic,   "Harmonic Order",       "Harmonic order N for this doubler; set once at profile creation",
     2,    1,          QVariant{}, HwSettingPriority::Required},
    {motorNumber,"Motor Number",         "Autotracker motor index (1-3) driving this doubling crystal",
     1,    1,          3,          HwSettingPriority::Important},
    {calScheme,  "Calibration Scheme",   "Tuning-curve model used to evaluate the doubling crystal's "
                                          "wavelength <-> motor position mapping",
     QVariant::fromValue(Scheme::Physical), QVariant{}, QVariant{}, HwSettingPriority::Important},
    {calCrystal, "Crystal Type",         "Doubling-crystal species; picks the Sellmeier equations for "
                                          "the Physical calibration scheme",
     QVariant::fromValue(CrystalType::BBO), QVariant{}, QVariant{}, HwSettingPriority::Important,
     {}, calScheme, QVariant::fromValue(Scheme::Physical)},
    {calInvert,  "Invert Phase Match",   "Select the alternate (-) branch of the Physical scheme's "
                                          "phase-match relation",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional,
     {}, calScheme, QVariant::fromValue(Scheme::Physical)}
)

// Sine-bar drive mechanics for the doubling-crystal motor stage. cutAngleDeg
// and temperature feed the Physical calibration scheme; the remaining
// entries are the sine-bar geometry and motor-control parameters shared by
// every scheme.
REGISTER_HARDWARE_ARRAY(SirahFcu, stages,
    "Crystal Stage Geometry", "Sine-bar tuning geometry for the doubling-crystal motor stage",
    HwSettingPriority::Important, calScheme, QVariant::fromValue(Scheme::Physical))
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
// row per order. Registered with no entries; populated by CSV import. The
// column schema is registered separately (REGISTER_HARDWARE_ARRAY_SCHEMA) so
// HwArrayEditDialog still has the right columns before the first import.
REGISTER_HARDWARE_ARRAY(SirahFcu, polyCoeffs,
    "Polynomial Coefficients", "Forward/inverse coefficient lists for the Polynomial calibration scheme",
    HwSettingPriority::Optional, calScheme, QVariant::fromValue(Scheme::Polynomial))
REGISTER_HARDWARE_ARRAY_SCHEMA(SirahFcu, polyCoeffs, pcOrder, pcForward, pcInverse)

// Spline calibration scheme: imported (wavelength, position) point table.
// Registered with no entries; populated by CSV import. Column schema
// registered separately, as for polyCoeffs above.
REGISTER_HARDWARE_ARRAY(SirahFcu, splinePoints,
    "Spline Points", "Wavelength/position point table for the Spline calibration scheme",
    HwSettingPriority::Optional, calScheme, QVariant::fromValue(Scheme::Spline))
REGISTER_HARDWARE_ARRAY_SCHEMA(SirahFcu, splinePoints, spWavelength, spPosition)

SirahFcu::SirahFcu(const QString& label, QObject *parent) :
    LaserFreqConversionStage(QString(SirahFcu::staticMetaObject.className()), label, parent)
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
    // The Autotracker has no auto-prompt stream to silence (unlike the
    // Cobra's 1 Hz status broadcast); establish the connection with an
    // Identify round-trip instead.
    auto idCmd = BC::Autotracker::buildCommand(0x02);
    p_comm->writeBinary(idCmd);
    auto resp = p_comm->readBytes(12,true);

    BC::Autotracker::Response r;
    if(!BC::Autotracker::parseResponse(resp, r))
    {
        hwError(u"No response to Identify command (Hex: %1)."_s.arg(QString(resp.toHex())));
        return false;
    }

    if(r.id == 0x01 && r.payload.size() >= 3)
        hwDebug(u"Autotracker ROM version %1, revision %2."_s
                    .arg(static_cast<quint8>(r.payload.at(1)))
                    .arg(static_cast<quint8>(r.payload.at(2))));

    // Drain any error codes left queued from a previous session before
    // trusting a subsequent reply's Adr/Status byte or Error query.
    if(!drainErrorQueue())
        return false;

    // Reprogram the motor's move profile: the Autotracker holds it in
    // volatile state and has no defined profile for this motor after a power
    // cycle, so a move issued without one has no ramp to run.
    if(!setMoveParameters())
        return false;

    return prompt();
}

bool SirahFcu::setMoveParameters()
{
    // Set Command Move Parameters (0x1F) is the only way to establish the
    // motor's start/high frequency and ramp length; there is no documented
    // "acceleration undefined" Autotracker error code, so an unset profile
    // would surface only as a botched move rather than a clean fault.
    auto startHz = qBound(0.0, getArrayValue(stages,0,sStart,3000.0), 65535.0);
    auto highHz = qBound(0.0, getArrayValue(stages,0,sHigh,12000.0), 65535.0);
    auto ramp = qBound(0.0, static_cast<double>(getArrayValue(stages,0,sRamp,2400)), 65535.0);

    auto startF = static_cast<quint16>(qRound(startHz));
    auto highF = static_cast<quint16>(qRound(highHz));
    auto rampSteps = static_cast<quint16>(qRound(ramp));

    QByteArray dat;
    dat.append(static_cast<char>(motor()));
    dat.append(static_cast<char>((startF >> 8) & 0xFF));
    dat.append(static_cast<char>(startF & 0xFF));
    dat.append(static_cast<char>((highF >> 8) & 0xFF));
    dat.append(static_cast<char>(highF & 0xFF));
    dat.append(static_cast<char>((rampSteps >> 8) & 0xFF));
    dat.append(static_cast<char>(rampSteps & 0xFF));

    auto cmd = BC::Autotracker::buildCommand(0x1F, dat);
    p_comm->writeBinary(cmd);
    auto resp = p_comm->readBytes(12,true);

    BC::Autotracker::Response r;
    if(!BC::Autotracker::parseResponse(resp, r) || r.id != 0x00)
    {
        reportCommError(u"Could not set move parameters (start/high frequency, ramp length) for motor %1."_s.arg(motor()));
        return false;
    }

    return true;
}

quint8 SirahFcu::motor() const
{
    return static_cast<quint8>(qBound(1, get(motorNumber, 1), 3));
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
        // A config error, not a transient mismatch: it will not resolve
        // itself on the next point, so this must hard-fail regardless of
        // the verify flag rather than leaving the FCU silently parked while
        // the caller believes the move succeeded.
        hwError(u"Cannot set position to %1 cm-1 (%2 nm): %3"_s
                    .arg(localCm1,0,'f',3).arg(wl,0,'f',4).arg(d_calibration.errorString()));
        emit hardwareFailure();
        return;
    }

    if(!prompt())
    {
        // Could not even read the current status before attempting the
        // move; treat the same as a failed move rather than silently
        // leaving the motor at its previous position.
        hwError(u"Could not set position to %1 cm-1 (%2 nm)."_s
                    .arg(localCm1,0,'f',3).arg(wl,0,'f',4));
        emit hardwareFailure();
        return;
    }

    //calculate target position; guard against a Spline calibration's
    //out-of-domain NaN sentinel before the cast below, which would
    //otherwise be undefined behavior (typically INT_MIN, driving the motor
    //to a hard-stop with no error).
    auto rawTargetPos = d_calibration.wavelengthToPos(wl);
    if(!std::isfinite(rawTargetPos))
    {
        hwError(u"Wavelength %1 nm (%2 cm-1) is outside the calibration's valid range; refusing to move."_s
                    .arg(wl,0,'f',4).arg(localCm1,0,'f',3));
        return;
    }

    // Positions are unsigned 24-bit on the Autotracker (0..0xFFFFFF); guard
    // the cast below the same way the isfinite() check above guards against
    // a Spline calibration's NaN sentinel -- a negative or over-range
    // result cast to quint32 would otherwise wrap into a bogus in-range
    // position and silently drive the motor to the wrong place (or into a
    // hard-stop) rather than reporting a calibration/domain error.
    if(rawTargetPos < 0.0 || rawTargetPos > 16777215.0)
    {
        hwError(u"Calculated motor position %1 for %2 nm (%3 cm-1) is outside the Autotracker's "
                 "24-bit position range; refusing to move."_s
                    .arg(rawTargetPos,0,'f',1).arg(wl,0,'f',4).arg(localCm1,0,'f',3));
        return;
    }

    auto targetPos = static_cast<quint32>(qRound(rawTargetPos));
    auto currentPos = d_lastPos;
    auto delta = static_cast<qint32>(static_cast<qint64>(targetPos) - static_cast<qint64>(currentPos));

    // Skip a redundant move only when doing so cannot violate the verify
    // window. Rather than gating on a fixed step count (which a coarse
    // sine-bar pitch could translate to more than the verify tolerance),
    // convert the unmoved current position back to a wavenumber via the
    // same calibration used above and compare it directly to the request;
    // half the tolerance leaves headroom for the verify readback's own
    // rounding. A non-finite currentWl (e.g. current position outside the
    // calibration's domain) falls through to an actual move rather than
    // risking an unsafe skip.
    auto verifyToleranceCm1 = get(tolerance, 1.0);
    auto currentWl = d_calibration.posToWavelength(currentPos);
    if(std::isfinite(currentWl) && qAbs(toCm1(currentWl, LaserUnit::Nm) - localCm1) < 0.5*verifyToleranceCm1)
        return;

    //can we just move relative?
    //Conditions: need last move to be in same direction as backlash correction,
    //and distance should be less than backlash correction.
    auto backlash = getArrayValue(stages,0,sbls,24000);
    if(d_lastMoveDir != 0 && (d_lastMoveDir*delta) > 0 && qAbs(delta) < qAbs(backlash))
    {
        moveRelative(delta);
        if(backlash > 0)
            d_lastMoveDir = 1;
        else
            d_lastMoveDir = -1;
    }
    else
    {
        // The two-move backlash approach (move to target-backlash, then a
        // relative move by backlash) was written for the Cobra's signed
        // 32-bit position space, where target-backlash simply going
        // negative was harmless. Autotracker positions are unsigned
        // 24-bit, so that intermediate can underflow below 0 (or, for a
        // negative backlash setting, overflow past 0xFFFFFF). When the
        // approach point would fall outside the valid range, skip the
        // backlash pre-move entirely and go straight to the target; the
        // backlash direction is simply unknown after a direct move, so the
        // next call cannot use the single-relative-move shortcut above.
        auto approach = static_cast<qint64>(targetPos) - backlash;
        if(approach < 0 || approach > 0xFFFFFF)
        {
            moveAbsolute(targetPos);
            d_lastMoveDir = 0;
        }
        else if(moveAbsolute(static_cast<quint32>(approach)))
        {
            moveRelative(backlash);
            if(backlash > 0)
                d_lastMoveDir = 1;
            else
                d_lastMoveDir = -1;
        }
        else
            d_lastMoveDir = 0;
    }
}

double SirahFcu::readPos()
{
    if(!prompt())
    {
        hwError("Could not read position."_L1);
        return -1.0;
    }

    auto wl = d_calibration.posToWavelength(d_lastPos);
    if(!std::isfinite(wl))
    {
        hwError(u"Could not convert motor position %1 to a wavelength."_s.arg(d_lastPos));
        return -1.0;
    }

    return toCm1(wl, LaserUnit::Nm);
}

bool SirahFcu::prompt()
{
    auto m = motor();
    QByteArray dat(1, static_cast<char>(m));
    auto cmd = BC::Autotracker::buildCommand(0x17, dat);
    p_comm->writeBinary(cmd);
    auto resp = p_comm->readBytes(12,true);

    BC::Autotracker::Response r;
    if(!BC::Autotracker::parseResponse(resp, r) || r.id != 0x0b)
    {
        d_errorString = QString("Received unexpected response to Get Position (Hex: %1)").arg(QString(resp.toHex()));
        return false;
    }

    if(r.payload.isEmpty() || static_cast<quint8>(r.payload.at(0)) != m)
    {
        d_errorString = QString("Get Position echoed motor %1, expected %2.")
                             .arg(r.payload.isEmpty() ? -1 : static_cast<int>(static_cast<quint8>(r.payload.at(0))))
                             .arg(m);
        return false;
    }

    // Pos24 sits at payload bytes 1..3 (response bytes 4..6): payload byte 0
    // is the echoed motor number checked above.
    d_lastPos = BC::Autotracker::unpackPos24(r.payload, 1);
    return true;
}

void SirahFcu::moveRelative(qint32 steps)
{
    // TODO (bench-verify): the Goto Position frame carries no separate
    // direction byte for a relative move (unlike the Cobra's Move Relative,
    // which has distinct Dev/Dir/magnitude fields) -- only the 24-bit Pos
    // field itself. This encodes the signed delta as its 24-bit two's-
    // complement truncation, on the assumption that the controller
    // interprets Pos as signed when Rel=1. That assumption is not
    // bench-confirmed; §4.6 of the protocol reference notes the only
    // captured Goto exchange used Rel=0. If real hardware rejects this or
    // moves the wrong direction, the sign convention here is the first
    // thing to revisit.
    quint32 encoded = static_cast<quint32>(steps) & 0x00FFFFFFu;

    QByteArray dat;
    dat.append(static_cast<char>(motor()));
    dat.append(static_cast<char>(0x00)); // Wait = 0: ack only once the move completes
    dat.append(static_cast<char>(0x01)); // Rel = 1: relative to current position
    dat.append(BC::Autotracker::packPos24(encoded));

    auto cmd = BC::Autotracker::buildCommand(0x22, dat);
    p_comm->writeBinary(cmd);

    auto resp = readResponse(moveAckTimeoutMs);
    BC::Autotracker::Response r;
    if(!BC::Autotracker::parseResponse(resp, r) || r.id != 0x00)
    {
        reportCommError(u"Relative move by %1 steps did not complete successfully."_s.arg(steps));
        emit hardwareFailure();
    }
}

bool SirahFcu::moveAbsolute(quint32 targetPos)
{
    QByteArray dat;
    dat.append(static_cast<char>(motor()));
    dat.append(static_cast<char>(0x00)); // Wait = 0: ack only once the move completes
    dat.append(static_cast<char>(0x00)); // Rel = 0: absolute target
    dat.append(BC::Autotracker::packPos24(targetPos));

    auto cmd = BC::Autotracker::buildCommand(0x22, dat);
    p_comm->writeBinary(cmd);

    // Wait=0 means the controller withholds its acknowledgment until the
    // move physically completes; a full-travel move can take several
    // seconds, far longer than the short default comm timeout used for
    // ordinary queries. readResponse() polls p_comm in a bounded loop
    // rather than assuming a single read call can wait long enough, so a
    // genuinely wedged reply still times out instead of blocking forever.
    // There is no motor-running status bit to poll and no documented stop
    // command, so a move that does not ack within the timeout is simply
    // reported as failed -- the ack's payload is not a position field
    // (see autotrackerprotocol.h) and is not inspected.
    auto resp = readResponse(moveAckTimeoutMs);
    BC::Autotracker::Response r;
    if(!BC::Autotracker::parseResponse(resp, r) || r.id != 0x00)
    {
        reportCommError(u"Move to position %1 did not complete successfully."_s.arg(targetPos));
        emit hardwareFailure();
        return false;
    }

    return true;
}

QByteArray SirahFcu::readResponse(int totalTimeoutMs)
{
    QElapsedTimer timer;
    timer.start();

    QByteArray resp;
    do
    {
        resp = p_comm->readBytes(12,true);
        if(resp.size() == 12)
            return resp;
    } while(timer.elapsed() < totalTimeoutMs);

    return resp;
}

bool SirahFcu::drainErrorQueue()
{
    // The Error command reads a queue of stacked error codes that persists
    // across sessions until read; drain it once at connection time so a
    // stale queued error left over from an earlier session cannot later be
    // mistaken for a live fault (see autotrackerprotocol.h -- the Adr/Status
    // byte's error-flag bit is unreliable for exactly this reason). A
    // well-drained queue reports either an all-zero payload or a Stack
    // Underflow code (7, "no more entries"); the loop is bounded rather
    // than open-ended in case a malfunctioning unit never reports either.
    for(int i=0; i<8; i++)
    {
        auto cmd = BC::Autotracker::buildCommand(0x03);
        p_comm->writeBinary(cmd);
        auto resp = p_comm->readBytes(12,true);

        BC::Autotracker::Response r;
        if(!BC::Autotracker::parseResponse(resp, r))
        {
            hwError(u"No response to Error command while draining the startup error queue (Hex: %1)."_s
                        .arg(QString(resp.toHex())));
            return false;
        }

        bool anyNonzero = false;
        bool underflow = false;
        for(auto b : r.payload)
        {
            auto code = static_cast<quint8>(b);
            if(code == 0)
                continue;
            if(code == 7)
            {
                underflow = true;
                continue;
            }
            anyNonzero = true;
            hwDebug(u"Drained queued Autotracker error: %1"_s.arg(BC::Autotracker::errorString(code)));
        }

        if(!anyNonzero || underflow)
            break;
    }

    return true;
}

void SirahFcu::reportCommError(const QString &context)
{
    auto cmd = BC::Autotracker::buildCommand(0x03);
    p_comm->writeBinary(cmd);
    auto resp = p_comm->readBytes(12,true);

    BC::Autotracker::Response r;
    if(BC::Autotracker::parseResponse(resp, r))
    {
        QStringList codes;
        for(auto b : r.payload)
        {
            auto code = static_cast<quint8>(b);
            if(code != 0)
                codes << BC::Autotracker::errorString(code);
        }

        if(!codes.isEmpty())
        {
            hwError(u"%1 (%2)"_s.arg(context, codes.join(u"; "_s)));
            return;
        }
    }

    hwError(context);
}
