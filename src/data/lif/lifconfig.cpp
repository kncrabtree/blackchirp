#include <data/lif/lifconfig.h>

#include <data/lif/liftrace.h>
#include <data/loghandler.h>
#include <data/storage/blackchirpcsv.h>
#include <data/storage/enumcsvconvert.h>
#include <QDir>
#include <QFile>
#include <QRandomGenerator>
#include <QSaveFile>
#include <QTextStream>
#include <optional>
#include <set>

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

void LifConfig::setConversionNodes(std::vector<BC::LifConv::Node> nodes, const QString &laserKey)
{
    d_conversionNodes = std::move(nodes);
    d_conversionLaserKey = laserKey;

    auto result = LifConversion::assemble(d_conversionNodes);
    d_conversion = result.ok ? result.conversion : LifConversion();
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

        // Per-node output beam: value = A*fundamental + B (cm-1), read
        // directly from the assembled conversion (fixed at assembly time).
        const auto [a,b] = d_conversion.stageOutputCoeffs(node.stageKey);

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

bool LifConfig::readTopologyFile()
{
    QDir dir(BlackchirpCSV::exptDir(d_number,d_path));
    QFile f(dir.absoluteFilePath(BC::CSV::lifTopologyFile));
    if(!f.exists())
        return true; // no topology file: identity case, not an error

    if(!f.open(QIODevice::ReadOnly|QIODevice::Text))
        return false;

    struct RawRow {
        QString stageKey;
        BC::LifConv::Op op;
        int n;
        bool isFinal;
        QString in0;
        QString in1;
    };

    BlackchirpCSV csv;
    std::vector<RawRow> rows;
    std::set<QString> stageKeys;

    while(!f.atEnd())
    {
        auto l = csv.readLine(f);
        if(l.isEmpty())
            continue;

        if(l.constFirst().toString() == "Index"_L1)
            continue;

        if(l.size() != 9)
            continue;

        bool ok = false;
        l.at(0).toInt(&ok);
        if(!ok)
            continue;

        RawRow row;
        row.stageKey = l.at(1).toString();
        row.op = BC::CSV::enumFromVariant<BC::LifConv::Op>(l.at(2),BC::LifConv::Op::NHG);
        bool nOk = false;
        int n = l.at(3).toString().toInt(&nOk);
        row.n = nOk ? n : 2;
        row.isFinal = QVariant(l.at(4)).toBool();
        row.in0 = l.at(5).toString();
        row.in1 = l.at(6).toString();
        // Columns 7/8 (OutCoeffA/B) are derived data, recomputed via
        // assemble() below rather than trusted from disk.

        stageKeys.insert(row.stageKey);
        rows.push_back(std::move(row));
    }

    if(rows.empty())
    {
        setConversionNodes({},QString());
        return true;
    }

    // Classify one input token per the reader-notes contract: Fixed:<cm1> ->
    // Fixed; a token matching another row's StageKey -> Stage; anything
    // else -> Laser (and that token IS the laser hwKey the writer emitted).
    QString laserKey;
    auto classify = [&](const QString &tok) -> std::optional<BC::LifConv::InputRef>
    {
        if(tok.isEmpty())
            return std::nullopt;

        BC::LifConv::InputRef ref;
        if(tok.startsWith(u"Fixed:"_s))
        {
            ref.type = BC::LifConv::RefType::Fixed;
            ref.fixedCm1 = tok.mid(6).toDouble();
        }
        else if(stageKeys.count(tok))
        {
            ref.type = BC::LifConv::RefType::Stage;
            ref.stageKey = tok;
        }
        else
        {
            ref.type = BC::LifConv::RefType::Laser;
            laserKey = tok;
        }
        return ref;
    };

    std::vector<BC::LifConv::Node> nodes;
    nodes.reserve(rows.size());
    for(const auto &row : rows)
    {
        BC::LifConv::Node node;
        node.stageKey = row.stageKey;
        node.op = row.op;
        node.n = row.n;
        node.isFinal = row.isFinal;

        if(auto in0 = classify(row.in0))
            node.inputs.push_back(*in0);
        if(auto in1 = classify(row.in1))
            node.inputs.push_back(*in1);

        nodes.push_back(std::move(node));
    }

    setConversionNodes(std::move(nodes),laserKey);
    return true;
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
    // Serialize laser positions with d_laserDecimals fractional digits so a
    // future reader can recover the display precision via
    // peekValueString/countFractionalDigits without a dedicated header field.
    // The unit sits in column 6 of the same row and is read back the same
    // way. d_laserPosStart/Step are already in the display LaserUnit, so no
    // conversion is needed here. d_laserDecimals is the laser's display-
    // decimals setting, the same one that quantizes the start/step spin boxes
    // the axis is built from (ExperimentTypePage), so a value from that path
    // always round-trips at this width.
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
    bool laserUnitRecognized = laserUnitCell.isEmpty(); // empty cell -> silent Nm default
    for(auto u : {BC::LifConv::LaserUnit::Cm1, BC::LifConv::LaserUnit::Nm,
                  BC::LifConv::LaserUnit::GHz, BC::LifConv::LaserUnit::eV})
    {
        if(laserUnitCell == BC::LifConv::unitLabel(u))
        {
            d_laserUnits = u;
            laserUnitRecognized = true;
            break;
        }
    }
    if(!laserUnitRecognized)
        bcWarn(u"LifConfig: unrecognized laser unit \"%1\" in header.csv; defaulting to nm."_s.arg(laserUnitCell));

    const QString startValueStr = peekValueString(lStart);
    const QString stepValueStr = peekValueString(lStep);
    // Distinguish "inferred 0 fractional digits" (an integer-formatted
    // axis, e.g. "200") from "nothing to infer from" (both cells empty,
    // e.g. no LaserStart/LaserStep row was ever written): only the latter
    // should leave d_laserDecimals at its stale/default seed.
    if(!startValueStr.isEmpty() || !stepValueStr.isEmpty())
        d_laserDecimals = qMax(countFractionalDigits(startValueStr), countFractionalDigits(stepValueStr));
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
