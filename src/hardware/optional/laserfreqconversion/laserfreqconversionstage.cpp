#include <hardware/optional/laserfreqconversion/laserfreqconversionstage.h>

#include <hardware/core/hardwareregistration.h>
#include <data/storage/enumcsvconvert.h>
#include <data/loadout/loadoutmanager.h>

using namespace BC::Key::LaserConvStage;
using namespace BC::LifConv;

REGISTER_HARDWARE_BASE(LaserFreqConversionStage,
    {op,       "Conversion Operation", "Node operation: NHG (N-th harmonic), SFG, or DFG",
     QVariant::fromValue(Op::NHG), QVariant{}, QVariant{}, HwSettingPriority::Important},
    {harmonic, "Harmonic Order",      "Harmonic order N for an NHG node (ignored for SFG/DFG)",
     2, 1, QVariant{}, HwSettingPriority::Important},
    {verify,   "Verify Move",         "Confirm the achieved position after a move; a mismatch "
                                       "fails the move instead of only logging a warning",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {tolerance,"Verify Tolerance (cm-1)", "Move is verified when the read-back is within this "
                                       "many cm-1 of the requested local input wavenumber",
     1.0, 0.0, QVariant{}, HwSettingPriority::Optional}
)

LaserFreqConversionStage::LaserFreqConversionStage(const QString& impl, const QString& label, QObject *parent) :
    HardwareObject(QString(LaserFreqConversionStage::staticMetaObject.className()), impl, label, parent)
{
    d_threaded = true;
}

LaserFreqConversionStage::~LaserFreqConversionStage()
{

}

BC::LifConv::Op LaserFreqConversionStage::conversionOp() const
{
    return BC::CSV::enumFromVariant<Op>(get(op, QVariant::fromValue(Op::NHG)), Op::NHG);
}

bool LaserFreqConversionStage::setHarmonicOrder(int n)
{
    set(harmonic, n);
    save();
    return true;
}

double LaserFreqConversionStage::readPosition()
{
    return readPos();
}

bool LaserFreqConversionStage::setPosition(double localCm1)
{
    setPos(localCm1);
    double achieved = readPos();

    // A negative achieved is readPos()'s hard comm-error sentinel, not a
    // value the stage actually reported, so it must fail the move outright:
    // unlike an in-range readback that simply misses the requested
    // wavenumber, this is not something the verify flag should be able to
    // downgrade to a warning, or a dead stage would report success on every
    // point with verify off.
    if(achieved < 0.0)
    {
        auto msg = u"Move to %1 cm-1 could not be read back (communication error)."_s
                       .arg(localCm1,0,'f',3);
        hwError(msg);
        emit hardwareFailure();
        return false;
    }

    // The verification window is a per-device registered setting: phase-match
    // motors resolve to well under the 1 cm-1 default across the pipeline's
    // operating range (grating fundamentals span 5000-40000 cm-1, per
    // LifLaser::minPos/maxPos), but a coarser mount can widen it. The window
    // absorbs encoder/rounding noise while still catching a genuinely missed
    // move.
    auto verifyToleranceCm1 = get(tolerance, 1.0);

    bool matched = qAbs(achieved - localCm1) <= verifyToleranceCm1;
    if(matched)
        return true;

    auto msg = u"Move to %1 cm-1 could not be verified (read back %2 cm-1)."_s
                   .arg(localCm1,0,'f',3).arg(achieved,0,'f',3);

    if(get(verify, true))
    {
        hwError(msg);
        emit hardwareFailure();
        return false;
    }

    hwWarn(msg);
    return true;
}

std::vector<BC::LifConv::Node> lifConversionNodesFromSnapshot(const LifConversionSnapshot &snap)
{
    // op/harmonic are read from a SettingsStorage snapshot constructed
    // directly on each wiring entry's stage key, never a live device: the
    // caller may be running on the GUI/data-layer thread, and a stage's
    // conversionOp()/harmonicOrder() are virtuals on a threaded HardwareObject.
    auto opOf = [](const QString &stageKey) -> Op {
        SettingsStorage s(stageKey, SettingsStorage::Hardware);
        return BC::CSV::enumFromVariant<Op>(s.get(op, QVariant::fromValue(Op::NHG)), Op::NHG);
    };
    auto harmonicOf = [](const QString &stageKey) -> int {
        SettingsStorage s(stageKey, SettingsStorage::Hardware);
        return s.get(harmonic, 2);
    };

    return snap.toNodes(opOf, harmonicOf);
}

LifConversion::AssemblyResult assembleLifConversion(const LifConversionSnapshot &snap)
{
    return LifConversion::assemble(lifConversionNodesFromSnapshot(snap));
}

LifConversion::AssemblyResult assembleCurrentLifConversion()
{
    auto loadoutName = LoadoutManager::instance().currentLoadoutName();
    auto preset = LoadoutManager::instance().currentLifPreset(loadoutName);
    if(!preset)
    {
        // No loadout/LIF preset selected: identity, not an error, matching
        // the tolerant fallback GUI callers rely on to stay responsive
        // before a topology has been configured.
        LifConversion::AssemblyResult identity;
        identity.ok = true;
        identity.conversion = LifConversion();
        return identity;
    }

    return assembleLifConversion(preset->conversion);
}
