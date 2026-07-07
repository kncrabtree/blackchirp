#include <hardware/core/liflaser/liffreqconversionstage.h>

#include <hardware/core/hardwareregistration.h>
#include <data/storage/enumcsvconvert.h>

using namespace BC::Key::LifConvStage;
using namespace BC::LifConv;

REGISTER_HARDWARE_BASE(LifFreqConversionStage,
    {op,       "Conversion Operation", "Node operation: NHG (N-th harmonic), SFG, or DFG",
     QVariant::fromValue(Op::NHG), QVariant{}, QVariant{}, HwSettingPriority::Important},
    {harmonic, "Harmonic Order",      "Harmonic order N for an NHG node (ignored for SFG/DFG)",
     2, 1, QVariant{}, HwSettingPriority::Important},
    {isFinal,  "Final Beam",          "This node's output is the LIF excitation (output) beam",
     false, QVariant{}, QVariant{}, HwSettingPriority::Important},
    {verify,   "Verify Move",         "Confirm the achieved position after a move; a mismatch "
                                       "fails the move instead of only logging a warning",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {tolerance,"Verify Tolerance (cm-1)", "Move is verified when the read-back is within this "
                                       "many cm-1 of the requested local input wavenumber",
     1.0, 0.0, QVariant{}, HwSettingPriority::Optional}
)
REGISTER_HARDWARE_BASE_ARRAY(LifFreqConversionStage, inputs,
    "Conversion Inputs", "Ordered input references for this node (NHG needs one; SFG/DFG need two)",
    HwSettingPriority::Important)
REGISTER_HARDWARE_BASE_ARRAY_ENTRY(LifFreqConversionStage, inputs,
    {{refType, QVariant::fromValue(RefType::Laser)}, {refKey, QString()}, {refFixedCm1, 0.0}})

LifFreqConversionStage::LifFreqConversionStage(const QString& impl, const QString& label, QObject *parent) :
    HardwareObject(QString(LifFreqConversionStage::staticMetaObject.className()), impl, label, parent)
{
    d_threaded = true;
}

LifFreqConversionStage::~LifFreqConversionStage()
{

}

BC::LifConv::Node LifFreqConversionStage::conversionNode() const
{
    Node node;
    node.stageKey = d_key;
    node.op = BC::CSV::enumFromVariant<Op>(get(op, QVariant::fromValue(Op::NHG)), Op::NHG);
    node.n = get(harmonic, 2);
    node.isFinal = get(isFinal, false);

    auto count = getArraySize(inputs);
    for(std::size_t i=0; i<count; ++i)
    {
        InputRef ref;
        ref.type = BC::CSV::enumFromVariant<RefType>(
                    getArrayValue(inputs, i, refType, QVariant::fromValue(RefType::Laser)), RefType::Laser);
        ref.stageKey = getArrayValue(inputs, i, refKey, QString());
        ref.fixedCm1 = getArrayValue(inputs, i, refFixedCm1, 0.0);
        node.inputs.push_back(ref);
    }

    return node;
}

double LifFreqConversionStage::readPosition()
{
    return readPos();
}

bool LifFreqConversionStage::setPosition(double localCm1)
{
    setPos(localCm1);
    double achieved = readPos();

    // The verification window is a per-device registered setting: phase-match
    // motors resolve to well under the 1 cm-1 default across the pipeline's
    // operating range (grating fundamentals span 5000-40000 cm-1, per
    // LifLaser::minPos/maxPos), but a coarser mount can widen it. The window
    // absorbs encoder/rounding noise while still catching a genuinely missed
    // move.
    auto verifyToleranceCm1 = get(tolerance, 1.0);

    bool matched = achieved >= 0.0 && qAbs(achieved - localCm1) <= verifyToleranceCm1;
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
