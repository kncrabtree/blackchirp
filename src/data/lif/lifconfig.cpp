#include <data/lif/lifconfig.h>

#include <data/lif/liftrace.h>
#include <data/storage/blackchirpcsv.h>
#include <QDir>
#include <QFile>
#include <QRandomGenerator>
#include <QSaveFile>
#include <QTextStream>
#include <cmath>

using namespace Qt::Literals::StringLiterals;


LifConfig::LifConfig(const QString& digitizerHwKey) : HeaderStorage(BC::Store::LIF::key)
{
    ps_digitizerConfig = std::make_shared<LifDigitizerConfig>(digitizerHwKey);
}

void LifConfig::setLaserUnits(BC::LifConv::LaserUnit units)
{
    d_laserUnits = units;
}

void LifConfig::setLaserDecimals(int decimals)
{
    d_laserDecimals = qMax(0, decimals);
}

void LifConfig::setConversionTopology(const std::vector<BC::LifConv::Node> &nodes,
                                      const LifConversion &conv,
                                      const QString &laserKey)
{
    d_conversionNodes = nodes;
    d_conversion = conv;
    d_conversionLaserKey = laserKey;
}

bool LifConfig::writeTopologyFile() const
{
    // Identity / bare-laser case: the output beam is the grating fundamental
    // and header.csv already records the full display-unit axis, so a
    // topology file would add nothing.
    if(d_conversionNodes.empty())
        return true;

    QDir dir(BlackchirpCSV::exptDir(d_number,d_path));
    QSaveFile f(dir.absoluteFilePath(BC::CSV::lifTopologyFile));
    if(!f.open(QIODevice::WriteOnly|QIODevice::Text))
        return false;

    // Compact self-describing token for one ordered input: the tunable
    // source ("Laser"), a fixed mixing beam ("Fixed:<cm-1>"), or another
    // node's output (that stage's hwKey).
    auto refToken = [this](const BC::LifConv::InputRef &r) -> QVariant {
        switch(r.type)
        {
        case BC::LifConv::RefType::Laser:
            // The tunable source is the active LifLaser; identify it by its
            // real hwKey (fall back to a sentinel only if unknown).
            return d_conversionLaserKey.isEmpty() ? QVariant(u"Laser"_s)
                                                   : QVariant(d_conversionLaserKey);
        case BC::LifConv::RefType::Fixed:
            return u"Fixed:%1"_s.arg(r.fixedCm1,0,'f',6);
        case BC::LifConv::RefType::Stage:
            return r.stageKey;
        }
        return QString();
    };

    QTextStream t(&f);
    BlackchirpCSV::writeLine(t,{"Index","StageKey","Op","Harmonic","IsFinal",
                                "Input0","Input1","OutCoeffA","OutCoeffB"});
    for(std::size_t i=0; i<d_conversionNodes.size(); ++i)
    {
        const auto &node = d_conversionNodes.at(i);

        // Per-node output beam: value = A*fundamental + B (cm-1). Recover the
        // slope/intercept from two evaluations of the assembled conversion.
        const double b = d_conversion.stageOutput(node.stageKey,0.0);
        const double a = d_conversion.stageOutput(node.stageKey,1.0) - b;

        BlackchirpCSV::writeLine(t,{
            static_cast<int>(i),
            node.stageKey,
            QVariant::fromValue(node.op),
            node.op == BC::LifConv::Op::NHG ? QVariant(node.n) : QVariant(QString()),
            node.isFinal,
            node.inputs.size() > 0 ? refToken(node.inputs[0]) : QVariant(QString()),
            node.inputs.size() > 1 ? refToken(node.inputs[1]) : QVariant(QString()),
            a,
            b
        });
    }
    return f.commit();
}

namespace {
int countFractionalDigits(const QString& cell)
{
    const auto dot = cell.indexOf(QLatin1Char('.'));
    if(dot < 0)
        return 0;
    auto tail = cell.mid(dot + 1);
    // Stop at the first non-digit; scientific-notation exponents and
    // stray characters don't count toward display precision.
    int n = 0;
    for(QChar c : tail)
    {
        if(c.isDigit())
            ++n;
        else
            break;
    }
    return n;
}
}

bool LifConfig::isComplete() const
{
    return d_complete;
}

double LifConfig::currentDelay() const
{
    return static_cast<double>(d_currentDelayIndex)*d_delayStepUs + d_delayStartUs;
}

double LifConfig::currentLaserPos() const
{
    // The scan grid (d_laserPosStart/Step) is uniform in the display
    // LaserUnit (contract §F): some lasers actuate only in a native unit
    // at a fixed resolution, and a uniform-cm⁻¹ grid would round to an
    // uneven step sequence in that unit. This is the single point where
    // the axis crosses into output-beam cm⁻¹ for hardware dispatch.
    auto displayPos = static_cast<double>(d_currentLaserIndex)*d_laserPosStep + d_laserPosStart;
    return BC::LifConv::toCm1(displayPos, d_laserUnits);
}

QPair<double, double> LifConfig::delayRange() const
{
    return qMakePair(d_delayStartUs,d_delayStartUs + d_delayStepUs*(d_delayPoints-1));
}

QPair<double, double> LifConfig::laserRange() const
{
    return qMakePair(d_laserPosStart,d_laserPosStart + d_laserPosStep*(d_laserPosPoints-1));
}

int LifConfig::targetShots() const
{
    return d_delayPoints*d_laserPosPoints*d_shotsPerPoint;
}

int LifConfig::completedShots() const
{
    return ps_storage->completedShots();
}

QPair<int, int> LifConfig::lifGate() const
{
    return {d_procSettings.lifGateStart,d_procSettings.lifGateEnd};
}

QPair<int, int> LifConfig::refGate() const
{
    return {d_procSettings.refGateStart,d_procSettings.refGateEnd};
}

void LifConfig::addWaveform(const QVector<qint8> d)
{
    //the boolean returned by this function tells if the point was incremented
    if(d_complete && d_completeMode == StopWhenComplete)
        return;

    LifTrace t(digitizerConfig(),d,d_currentDelayIndex,d_currentLaserIndex);
    ps_storage->addTrace(t);
}

void LifConfig::loadLifData()
{
    ps_storage = std::make_shared<LifStorage>(d_delayPoints,d_laserPosPoints,d_number,d_path);
    LifTrace::LifProcSettings s;
    if(ps_storage->readProcessingSettings(s))
        d_procSettings = s;
    ps_storage->finish();
}

void LifConfig::storeValues()
{
    using namespace BC::Store::LIF;
    store(order,d_order);
    store(completeMode,d_completeMode);
    store(dStart,d_delayStartUs,BC::Unit::us);
    store(dStep,d_delayStepUs,BC::Unit::us);
    store(dPoints,d_delayPoints);
    // Serialize laser positions with locked fractional-digit count so a
    // future reader can recover the display precision via peekValueString
    // / countFractionalDigits without a dedicated header field. The unit
    // sits in column 6 of the same row and is read back the same way.
    // d_laserPosStart/Step are already in the display LaserUnit, so no
    // conversion is needed here.
    const auto laserUnitStr = BC::LifConv::unitLabel(d_laserUnits);
    store(lStart,QString::number(d_laserPosStart,'f',d_laserDecimals),laserUnitStr);
    store(lStep,QString::number(d_laserPosStep,'f',d_laserDecimals),laserUnitStr);
    store(dRandom,d_delayRandom);
    store(lPoints,d_laserPosPoints);
    store(shotsPerPoint,d_shotsPerPoint);

}

void LifConfig::retrieveValues()
{
    using namespace BC::Store::LIF;
    d_order = retrieve(order,DelayFirst);
    d_completeMode = retrieve(completeMode,ContinueAveraging);
    d_delayStartUs = retrieve(dStart,0.0);
    d_delayStepUs = retrieve(dStep,0.0);
    d_delayPoints = retrieve(dPoints,0);
    d_delayRandom = retrieve(dRandom,false);
    // Peek the laser unit cell and the raw value strings before
    // retrieve() consumes the row. Units come from column 6 of the
    // LaserStart row; decimals are inferred from the maximum fractional
    // digit count across LaserStart and LaserStep.
    const auto laserUnitCell = peekUnit(lStart);
    d_laserUnits = BC::LifConv::LaserUnit::Nm;
    for(auto u : {BC::LifConv::LaserUnit::Cm1, BC::LifConv::LaserUnit::Nm,
                  BC::LifConv::LaserUnit::GHz, BC::LifConv::LaserUnit::eV})
    {
        if(laserUnitCell == BC::LifConv::unitLabel(u))
        {
            d_laserUnits = u;
            break;
        }
    }
    const int startDecimals = countFractionalDigits(peekValueString(lStart));
    const int stepDecimals  = countFractionalDigits(peekValueString(lStep));
    const int inferred = qMax(startDecimals, stepDecimals);
    if(inferred > 0)
        d_laserDecimals = inferred;
    d_laserPosStart = retrieve(lStart,0.0);
    d_laserPosStep = retrieve(lStep,0.0);
    d_laserPosPoints = retrieve(lPoints,0);
    d_shotsPerPoint = retrieve(shotsPerPoint,0);

}

void LifConfig::prepareChildren()
{
    addChild(&digitizerConfig());
}


bool LifConfig::initialize()
{
    ps_storage = std::make_shared<LifStorage>(d_delayPoints,d_laserPosPoints,d_number,d_path);
    d_delayIndices.clear();
    for(int i=0; i<d_delayPoints; i++)
        d_delayIndices.append(i);
    if(d_delayRandom)
        std::shuffle(d_delayIndices.begin(),d_delayIndices.end(),*QRandomGenerator::global());
    d_delayScanIndex = 0;
    d_currentDelayIndex = d_delayIndices[0];
    ps_storage->writeProcessingSettings(d_procSettings);
    ps_storage->start();
    d_processingPaused = true;
    return true;
}

bool LifConfig::advance()
{
    //return true if we have enough shots for this point on this pass
    int c = ps_storage->currentTraceShots();
    int target = d_shotsPerPoint*(d_completedSweeps+1);

    bool inc = (c>=target);
    if(inc)
    {
        d_processingPaused = true;

        //if we have completed a delay sweep, randomize if needed
        if( (d_delayScanIndex + 1 >= d_delayPoints) && d_delayRandom )
            std::shuffle(d_delayIndices.begin(),d_delayIndices.end(),*QRandomGenerator::global());

        if(d_delayScanIndex+1 >= d_delayPoints && d_currentLaserIndex+1 >= d_laserPosPoints)
        {
            d_completedSweeps++;
            d_complete = true;
        }

        if(d_order == LaserFirst)
        {
            if(d_currentLaserIndex+1 >= d_laserPosPoints)
            {
                d_delayScanIndex = (d_delayScanIndex+1)%d_delayPoints;
                d_currentDelayIndex = d_delayIndices[d_delayScanIndex];
            }

            d_currentLaserIndex = (d_currentLaserIndex+1)%d_laserPosPoints;
        }
        else
        {
            if(d_delayScanIndex+1 >= d_delayPoints)
                d_currentLaserIndex = (d_currentLaserIndex+1)%d_laserPosPoints;

            d_delayScanIndex = (d_delayScanIndex+1)%d_delayPoints;
            d_currentDelayIndex = d_delayIndices[d_delayScanIndex];
        }
        ps_storage->advance();
    }
    return inc;
}

void LifConfig::hwReady()
{
    d_processingPaused = false;
}

int LifConfig::perMilComplete() const
{
    auto i = completedShots();
    return qBound(0,(1000*i)/(d_shotsPerPoint*d_delayPoints*d_laserPosPoints),1000);
}

bool LifConfig::indefinite() const
{
    if(d_completeMode == ContinueAveraging)
        return perMilComplete() >= 1000;

    return false;
}

bool LifConfig::abort()
{
    return false;
}

QString LifConfig::objectiveKey() const
{
    return BC::Config::Exp::lifType;
}


void LifConfig::cleanupAndSave()
{
    ps_storage->finish();
    ps_storage->save();
}
