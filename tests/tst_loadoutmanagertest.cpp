#include <QtTest>
#include <QCoreApplication>
#include <QSettings>
#include <QTemporaryDir>
#include <QDebug>

#include <src/data/loadout/loadoutmanager.h>
#include <src/data/loadout/hardwareloadout.h>

using namespace BC::Loadout;
using namespace BC::Store::LM;

// Matches the friend declaration in loadoutmanager.h
class LoadoutManagerTest : public QObject
{
    Q_OBJECT

public:
    LoadoutManagerTest();
    ~LoadoutManagerTest() override;

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();

    void testRoundTripWithFtmwPresets();
    void testRoundTripNoFtmwPresets();
    void testRemoveLoadout();
    void testCurrentDefaultPersistence();
    void testClockArrayRoundTrip();
    void testDigitizerChannelArrayRoundTrip();
    void testLoadoutsMatchingHwKey();
    void testCopyClocksMatching();
    void testCopyRfScalars();
    void testFtmwPresetCrud();
    void testCurrentFtmwPresetPointers();
    void testRenameFtmwPresetRewritesPointers();
    void testRemoveLoadoutCascadesFtmwPresets();

    void testRoundTripWithLifPresets();
    void testLifPresetCrud();
    void testCurrentLifPresetPointers();
    void testRenameLifPresetRewritesPointers();
    void testRemoveLoadoutCascadesLifPresets();
    void testLifConversionWiringArrayRoundTrip();
    void testClearLifPresets();
    void testPutLoadoutPreservesLifPresetsAcrossResave();

    void testHardwareIdentityRoundTrip();
    void testHardwareIdentityMigration();
    void testFtmwPresetReferencesHardware();
    void testLifPresetReferencesHardware();

private:
    LoadoutManager *makeLm() const;

    static FtmwPreset makeFtmwPreset(const QString &digiHwKey);
    static HardwareLoadout makeHardwareOnly();
    static HardwareLoadout makeWithPresets();
    static void verifyClocksEqual(const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &a,
                                  const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &b);
    static void verifyPresetEqual(const FtmwPreset &a, const FtmwPreset &b);

    static LifPreset makeLifPreset(const QString &laserKey);
    static HardwareLoadout makeWithLifPresets();
    static void verifyLifPresetEqual(const LifPreset &a, const LifPreset &b);

    QTemporaryDir *d_tempDir{nullptr};
    static constexpr auto s_org = "CrabtreeLab";
    static constexpr auto s_app = "BlackchirpLoadoutTest";
};

LoadoutManagerTest::LoadoutManagerTest() {}
LoadoutManagerTest::~LoadoutManagerTest() {}

void LoadoutManagerTest::initTestCase()
{
    d_tempDir = new QTemporaryDir();
    QVERIFY(d_tempDir->isValid());
    qDebug() << d_tempDir->path();
    QCoreApplication::setOrganizationName(s_org);
    QCoreApplication::setApplicationName(s_app);
    // QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, d_tempDir->path());
}

void LoadoutManagerTest::cleanupTestCase()
{
    delete d_tempDir;
    d_tempDir = nullptr;
}

void LoadoutManagerTest::init()
{
    QSettings s;
    s.clear();
    s.sync();
}

LoadoutManager *LoadoutManagerTest::makeLm() const
{
    return new LoadoutManager(s_org, s_app);
}

FtmwPreset LoadoutManagerTest::makeFtmwPreset(const QString &digiHwKey)
{
    using namespace Qt::StringLiterals;

    FtmwPreset preset;
    preset.digiHwKey = digiHwKey;

    preset.rfConfig.commonUpDownLO  = true;
    preset.rfConfig.awgMult         = 2.5;
    preset.rfConfig.upMixSideband   = RfConfig::LowerSideband;
    preset.rfConfig.chirpMult       = 4.0;
    preset.rfConfig.downMixSideband = RfConfig::UpperSideband;

    {
        RfConfig::ClockFreq cf;
        cf.hwKey          = u"Clock.ref"_s;
        cf.output         = 1;
        cf.op             = RfConfig::Multiply;
        cf.factor         = 10.0;
        cf.desiredFreqMHz = 100.0;
        preset.rfConfig.clocks.insert(RfConfig::AwgRef, cf);
    }
    {
        RfConfig::ClockFreq cf;
        cf.hwKey          = u"Clock.lo"_s;
        cf.output         = 0;
        cf.op             = RfConfig::Divide;
        cf.factor         = 2.0;
        cf.desiredFreqMHz = 8500.0;
        preset.rfConfig.clocks.insert(RfConfig::UpLO, cf);
    }

    preset.chirpConfig.setAwgSampleRate(4e9);
    preset.chirpConfig.setNumChirps(2);
    preset.chirpConfig.setChirpInterval(500.0);
    preset.chirpConfig.addSegment(7000.0, 8000.0, 5.0, 0);
    preset.chirpConfig.addSegment(7000.0, 8000.0, 5.0, 1);

    QVector<MarkerChannel> markers;
    MarkerChannel mc;
    mc.name       = u"ProtectionGate"_s;
    mc.role       = MarkerRole::Protection;
    mc.timingMode = MarkerChannel::ChirpRelative;
    mc.startTime  = -0.25;
    mc.endTime    = 0.25;
    mc.enabled    = true;
    markers.push_back(mc);
    preset.chirpConfig.setMarkerChannels(markers);

    preset.digitizer = FtmwDigitizerConfig(digiHwKey);
    preset.digitizer.d_triggerChannel   = 2;
    preset.digitizer.d_triggerSlope     = DigitizerConfig::FallingEdge;
    preset.digitizer.d_triggerDelayUSec = 0.1;
    preset.digitizer.d_triggerLevel     = 0.5;
    preset.digitizer.d_sampleRate       = 4e9;
    preset.digitizer.d_recordLength     = 1024;
    preset.digitizer.d_bytesPerPoint    = 2;
    preset.digitizer.d_byteOrder        = DigitizerConfig::LittleEndian;
    preset.digitizer.d_blockAverage     = true;
    preset.digitizer.d_numAverages      = 100;
    preset.digitizer.d_multiRecord      = false;
    preset.digitizer.d_numRecords       = 1;
    preset.digitizer.d_fidChannel       = 3;

    DigitizerConfig::AnalogChannel ach;
    ach.enabled   = true;
    ach.fullScale = 0.1;
    ach.offset    = 0.02;
    preset.digitizer.d_analogChannels[1] = ach;

    DigitizerConfig::DigitalChannel dch;
    dch.enabled = true;
    dch.input   = false;
    dch.role    = 5;
    preset.digitizer.d_digitalChannels[0] = dch;

    return preset;
}

HardwareLoadout LoadoutManagerTest::makeHardwareOnly()
{
    using namespace Qt::StringLiterals;
    HardwareLoadout lo;
    lo.name = u"NoPresets"_s;
    lo.hardwareMap = {{u"AWG.main"_s, u"VirtualAwg"_s}};
    return lo;
}

HardwareLoadout LoadoutManagerTest::makeWithPresets()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout lo;
    lo.name = u"Alpha"_s;
    lo.hardwareMap = {
        {u"Clock.ref"_s,      u"VirtualClock"_s},
        {u"AWG.main"_s,       u"VirtualAwg"_s},
        {u"FtmwDigitizer.main"_s, u"VirtualDigitizer"_s},
    };

    lo.ftmwPresets[u"Primary"_s]   = makeFtmwPreset(u"FtmwDigitizer.main"_s);
    lo.ftmwPresets[u"Secondary"_s] = makeFtmwPreset(u"FtmwDigitizer.main"_s);
    lo.ftmwPresets[lastUsedFtmwPresetName.toString()] = makeFtmwPreset(u"FtmwDigitizer.main"_s);

    lo.currentFtmwPresetName = u"Secondary"_s;

    return lo;
}

void LoadoutManagerTest::verifyClocksEqual(
    const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &a,
    const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &b)
{
    QCOMPARE(a.size(), b.size());
    for (auto it = a.constBegin(); it != a.constEnd(); ++it) {
        QVERIFY(b.contains(it.key()));
        const auto &ca = it.value();
        const auto &cb = b[it.key()];
        QCOMPARE(ca.hwKey,          cb.hwKey);
        QCOMPARE(ca.output,         cb.output);
        QCOMPARE(ca.op,             cb.op);
        QCOMPARE(ca.factor,         cb.factor);
        QCOMPARE(ca.desiredFreqMHz, cb.desiredFreqMHz);
    }
}

void LoadoutManagerTest::verifyPresetEqual(const FtmwPreset &a, const FtmwPreset &b)
{
    QCOMPARE(a.digiHwKey, b.digiHwKey);

    QCOMPARE(a.rfConfig.commonUpDownLO,  b.rfConfig.commonUpDownLO);
    QCOMPARE(a.rfConfig.awgMult,         b.rfConfig.awgMult);
    QCOMPARE(a.rfConfig.upMixSideband,   b.rfConfig.upMixSideband);
    QCOMPARE(a.rfConfig.chirpMult,       b.rfConfig.chirpMult);
    QCOMPARE(a.rfConfig.downMixSideband, b.rfConfig.downMixSideband);
    verifyClocksEqual(a.rfConfig.clocks, b.rfConfig.clocks);

    QCOMPARE(a.chirpConfig.numChirps(),     b.chirpConfig.numChirps());
    QCOMPARE(a.chirpConfig.chirpInterval(), b.chirpConfig.chirpInterval());
    const auto &acl = a.chirpConfig.chirpList();
    const auto &bcl = b.chirpConfig.chirpList();
    QCOMPARE(acl.size(), bcl.size());
    for (int ci = 0; ci < acl.size(); ++ci) {
        QCOMPARE(acl[ci].size(), bcl[ci].size());
        for (int si = 0; si < acl[ci].size(); ++si) {
            QCOMPARE(acl[ci][si].startFreqMHz, bcl[ci][si].startFreqMHz);
            QCOMPARE(acl[ci][si].endFreqMHz,   bcl[ci][si].endFreqMHz);
            QCOMPARE(acl[ci][si].durationUs,   bcl[ci][si].durationUs);
        }
    }

    const auto &ad = a.digitizer;
    const auto &bd = b.digitizer;
    QCOMPARE(ad.d_triggerChannel,   bd.d_triggerChannel);
    QCOMPARE(ad.d_triggerSlope,     bd.d_triggerSlope);
    QCOMPARE(ad.d_sampleRate,       bd.d_sampleRate);
    QCOMPARE(ad.d_recordLength,     bd.d_recordLength);
    QCOMPARE(ad.d_blockAverage,     bd.d_blockAverage);
    QCOMPARE(ad.d_numAverages,      bd.d_numAverages);
    QCOMPARE(ad.d_fidChannel,       bd.d_fidChannel);
    QCOMPARE(ad.d_analogChannels.size(),  bd.d_analogChannels.size());
    QCOMPARE(ad.d_digitalChannels.size(), bd.d_digitalChannels.size());
}

LifPreset LoadoutManagerTest::makeLifPreset(const QString &laserKey)
{
    using namespace Qt::StringLiterals;
    using namespace BC::LifConv;

    StageWiring doubler;
    doubler.stageKey = u"LifFreqConversionStage.doubler"_s;
    doubler.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    doubler.isFinal = false;

    StageWiring tripler;
    tripler.stageKey = u"LifFreqConversionStage.tripler"_s;
    tripler.inputs = {InputRef{RefType::Stage, u"LifFreqConversionStage.doubler"_s, 0.0},
                      InputRef{RefType::Fixed, {}, 50.0}};
    tripler.isFinal = true;

    LifConversionSnapshot snap;
    snap.laserKey = laserKey;
    snap.wiring = {doubler, tripler};

    LifPreset preset;
    preset.conversion = snap;
    return preset;
}

HardwareLoadout LoadoutManagerTest::makeWithLifPresets()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout lo;
    lo.name = u"LifAlpha"_s;
    lo.hardwareMap = {
        {u"LifLaser.default"_s,                    u"VirtualLifLaser"_s},
        {u"LifFreqConversionStage.doubler"_s,       u"VirtualLifFreqConversionStage"_s},
        {u"LifFreqConversionStage.tripler"_s,       u"VirtualLifFreqConversionStage"_s},
    };

    lo.lifPresets[u"Primary"_s]   = makeLifPreset(u"LifLaser.default"_s);
    lo.lifPresets[u"Secondary"_s] = makeLifPreset(u"LifLaser.default"_s);
    lo.lifPresets[lastUsedLifPresetName.toString()] = makeLifPreset(u"LifLaser.default"_s);

    lo.currentLifPresetName = u"Secondary"_s;

    return lo;
}

void LoadoutManagerTest::verifyLifPresetEqual(const LifPreset &a, const LifPreset &b)
{
    QCOMPARE(a.conversion.laserKey, b.conversion.laserKey);
    QCOMPARE(a.conversion.wiring.size(), b.conversion.wiring.size());
    for (std::size_t i = 0; i < a.conversion.wiring.size(); ++i) {
        const auto &wa = a.conversion.wiring[i];
        const auto &wb = b.conversion.wiring[i];
        QCOMPARE(wa.stageKey, wb.stageKey);
        QCOMPARE(wa.isFinal, wb.isFinal);
        QCOMPARE(wa.inputs.size(), wb.inputs.size());
        for (std::size_t j = 0; j < wa.inputs.size(); ++j) {
            QCOMPARE(wa.inputs[j].type,     wb.inputs[j].type);
            QCOMPARE(wa.inputs[j].stageKey, wb.inputs[j].stageKey);
            QCOMPARE(wa.inputs[j].fixedCm1, wb.inputs[j].fixedCm1);
        }
    }
}

// ── test cases ────────────────────────────────────────────────────────────────

void LoadoutManagerTest::testRoundTripWithFtmwPresets()
{
    using namespace Qt::StringLiterals;
    const HardwareLoadout original = makeWithPresets();

    {
        std::unique_ptr<LoadoutManager> lm(makeLm());
        QVERIFY(lm->putLoadout(original));
    }

    std::unique_ptr<LoadoutManager> lm2(makeLm());
    QVERIFY(lm2->loadoutExists(original.name));

    const auto got = lm2->getLoadout(original.name);
    QVERIFY(got.has_value());
    QCOMPARE(got->name,        original.name);
    QCOMPARE(got->hardwareMap, original.hardwareMap);
    QCOMPARE(got->currentFtmwPresetName, original.currentFtmwPresetName);

    // All three presets present
    QCOMPARE(got->ftmwPresets.size(), std::size_t(3));
    QVERIFY(got->ftmwPresets.count(u"Primary"_s));
    QVERIFY(got->ftmwPresets.count(u"Secondary"_s));
    QVERIFY(got->ftmwPresets.count(lastUsedFtmwPresetName.toString()));

    verifyPresetEqual(got->ftmwPresets.at(u"Primary"_s),
                      original.ftmwPresets.at(u"Primary"_s));
    verifyPresetEqual(got->ftmwPresets.at(u"Secondary"_s),
                      original.ftmwPresets.at(u"Secondary"_s));
}

void LoadoutManagerTest::testRoundTripNoFtmwPresets()
{
    const HardwareLoadout original = makeHardwareOnly();

    {
        std::unique_ptr<LoadoutManager> lm(makeLm());
        QVERIFY(lm->putLoadout(original));
    }

    std::unique_ptr<LoadoutManager> lm2(makeLm());
    const auto got = lm2->getLoadout(original.name);
    QVERIFY(got.has_value());
    QCOMPARE(got->name,        original.name);
    QCOMPARE(got->hardwareMap, original.hardwareMap);
    QVERIFY(got->ftmwPresets.empty());
    QVERIFY(got->currentFtmwPresetName.isEmpty());
}

void LoadoutManagerTest::testRemoveLoadout()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout alpha, beta;
    alpha.name = u"Alpha"_s;
    beta.name  = u"Beta"_s;

    {
        std::unique_ptr<LoadoutManager> lm(makeLm());
        lm->putLoadout(alpha);
        lm->putLoadout(beta);
        lm->setCurrentLoadoutName(u"Alpha"_s);
        QVERIFY(lm->loadoutExists(u"Alpha"_s));
        QVERIFY(lm->loadoutExists(u"Beta"_s));

        QVERIFY(lm->removeLoadout(u"Alpha"_s));
        QVERIFY(!lm->loadoutExists(u"Alpha"_s));
        QVERIFY(lm->currentLoadoutName() != u"Alpha"_s);
    }

    std::unique_ptr<LoadoutManager> lm2(makeLm());
    QVERIFY(!lm2->loadoutExists(u"Alpha"_s));
    QVERIFY(lm2->loadoutExists(u"Beta"_s));
    QVERIFY(lm2->currentLoadoutName() != u"Alpha"_s);
}

void LoadoutManagerTest::testCurrentDefaultPersistence()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout a, b;
    a.name = u"SetA"_s;
    b.name = u"SetB"_s;

    {
        std::unique_ptr<LoadoutManager> lm(makeLm());
        lm->putLoadout(a);
        lm->putLoadout(b);
        lm->setCurrentLoadoutName(u"SetA"_s);
        lm->setDefaultLoadoutName(u"SetB"_s);
    }

    std::unique_ptr<LoadoutManager> lm2(makeLm());
    QCOMPARE(lm2->currentLoadoutName(), u"SetA"_s);
    QCOMPARE(lm2->defaultLoadoutName(), u"SetB"_s);
}

void LoadoutManagerTest::testClockArrayRoundTrip()
{
    using namespace Qt::StringLiterals;

    RfConfigSnapshot snap;
    snap.commonUpDownLO  = false;
    snap.awgMult         = 1.0;
    snap.upMixSideband   = RfConfig::UpperSideband;
    snap.chirpMult       = 1.0;
    snap.downMixSideband = RfConfig::UpperSideband;

    {
        RfConfig::ClockFreq cf;
        cf.hwKey          = u"Clock.awg-ref"_s;
        cf.output         = 2;
        cf.op             = RfConfig::Divide;
        cf.factor         = 5.0;
        cf.desiredFreqMHz = 250.0;
        snap.clocks.insert(RfConfig::AwgRef, cf);
    }
    {
        RfConfig::ClockFreq cf;
        cf.hwKey          = u"Clock.uplo"_s;
        cf.output         = 0;
        cf.op             = RfConfig::Multiply;
        cf.factor         = 3.0;
        cf.desiredFreqMHz = 9000.0;
        snap.clocks.insert(RfConfig::UpLO, cf);
    }
    {
        RfConfig::ClockFreq cf;
        cf.hwKey          = u"Clock.dig-ref"_s;
        cf.output         = 1;
        cf.op             = RfConfig::Multiply;
        cf.factor         = 1.0;
        cf.desiredFreqMHz = 10.0;
        snap.clocks.insert(RfConfig::DigRef, cf);
    }

    const auto scalars = rfConfigScalarsMap(snap);
    const auto clocks  = rfConfigClocksArray(snap);
    const RfConfigSnapshot back = rfConfigSnapshotFromMaps(scalars, clocks);

    verifyClocksEqual(back.clocks, snap.clocks);
}

void LoadoutManagerTest::testDigitizerChannelArrayRoundTrip()
{
    using namespace Qt::StringLiterals;

    FtmwDigitizerConfig cfg(u"FtmwDigitizer.test"_s);
    cfg.d_analogChannels[0]  = {true,  0.05, -0.01};
    cfg.d_analogChannels[1]  = {false, 0.5,   0.0};
    cfg.d_digitalChannels[0] = {true,  true,  2};
    cfg.d_digitalChannels[1] = {true,  false, 7};
    cfg.d_fidChannel = 1;

    const auto analog   = digitizerAnalogArray(cfg);
    const auto digital  = digitizerDigitalArray(cfg);
    const auto scalars  = digitizerScalarsMap(cfg);
    const FtmwDigitizerConfig back = ftmwDigitizerFromMaps(
        u"FtmwDigitizer.test"_s, scalars, analog, digital);

    QCOMPARE(back.d_analogChannels.size(),  cfg.d_analogChannels.size());
    for (const auto &[idx, ch] : cfg.d_analogChannels) {
        QVERIFY(back.d_analogChannels.count(idx));
        QCOMPARE(back.d_analogChannels.at(idx).enabled,   ch.enabled);
        QCOMPARE(back.d_analogChannels.at(idx).fullScale, ch.fullScale);
        QCOMPARE(back.d_analogChannels.at(idx).offset,    ch.offset);
    }

    QCOMPARE(back.d_digitalChannels.size(), cfg.d_digitalChannels.size());
    for (const auto &[idx, ch] : cfg.d_digitalChannels) {
        QVERIFY(back.d_digitalChannels.count(idx));
        QCOMPARE(back.d_digitalChannels.at(idx).enabled, ch.enabled);
        QCOMPARE(back.d_digitalChannels.at(idx).input,   ch.input);
        QCOMPARE(back.d_digitalChannels.at(idx).role,    ch.role);
    }

    QCOMPARE(back.d_fidChannel, cfg.d_fidChannel);
}

void LoadoutManagerTest::testLoadoutsMatchingHwKey()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout a, b, c;
    a.name = u"A"_s;
    a.hardwareMap = {{u"AWG.main"_s, u"VirtualAwg"_s}, {u"Clock.ref"_s, u"VirtualClock"_s}};
    b.name = u"B"_s;
    b.hardwareMap = {{u"AWG.main"_s, u"VirtualAwg"_s}};
    c.name = u"C"_s;
    c.hardwareMap = {{u"Clock.ref"_s, u"VirtualClock"_s}};

    std::unique_ptr<LoadoutManager> lm(makeLm());
    lm->putLoadout(a);
    lm->putLoadout(b);
    lm->putLoadout(c);

    const QStringList awgMatches = lm->loadoutsMatchingHwKey(u"AWG.main"_s);
    QVERIFY(awgMatches.contains(u"A"_s));
    QVERIFY(awgMatches.contains(u"B"_s));
    QVERIFY(!awgMatches.contains(u"C"_s));

    const QStringList clockMatches = lm->loadoutsMatchingHwKey(u"Clock.ref"_s);
    QVERIFY(clockMatches.contains(u"A"_s));
    QVERIFY(!clockMatches.contains(u"B"_s));
    QVERIFY(clockMatches.contains(u"C"_s));

    const QStringList noMatches = lm->loadoutsMatchingHwKey(u"FtmwDigitizer.main"_s);
    QVERIFY(noMatches.isEmpty());
}

void LoadoutManagerTest::testCopyClocksMatching()
{
    using namespace Qt::StringLiterals;

    RfConfigSnapshot source;
    {
        RfConfig::ClockFreq cf;
        cf.hwKey = u"Clock.awg-ref"_s;
        cf.desiredFreqMHz = 100.0;
        source.clocks.insert(RfConfig::AwgRef, cf);
    }
    {
        RfConfig::ClockFreq cf;
        cf.hwKey = u"Clock.lo"_s;
        cf.desiredFreqMHz = 9000.0;
        source.clocks.insert(RfConfig::UpLO, cf);
    }

    RfConfigSnapshot dest;
    {
        RfConfig::ClockFreq cf;
        cf.hwKey = u"Clock.other"_s;
        cf.desiredFreqMHz = 50.0;
        dest.clocks.insert(RfConfig::DigRef, cf);
    }

    const std::set<QString> allowed = {u"Clock.awg-ref"_s};
    copyClocksMatching(source, dest, allowed);

    QVERIFY(dest.clocks.contains(RfConfig::AwgRef));
    QCOMPARE(dest.clocks[RfConfig::AwgRef].hwKey, u"Clock.awg-ref"_s);
    QVERIFY(!dest.clocks.contains(RfConfig::UpLO));
    QVERIFY(dest.clocks.contains(RfConfig::DigRef));
    QCOMPARE(dest.clocks[RfConfig::DigRef].hwKey, u"Clock.other"_s);
}

void LoadoutManagerTest::testCopyRfScalars()
{
    RfConfigSnapshot source;
    source.commonUpDownLO  = true;
    source.awgMult         = 3.14;
    source.upMixSideband   = RfConfig::LowerSideband;
    source.chirpMult       = 0.5;
    source.downMixSideband = RfConfig::LowerSideband;

    RfConfigSnapshot dest;
    copyRfScalars(source, dest);

    QCOMPARE(dest.commonUpDownLO,  source.commonUpDownLO);
    QCOMPARE(dest.awgMult,         source.awgMult);
    QCOMPARE(dest.upMixSideband,   source.upMixSideband);
    QCOMPARE(dest.chirpMult,       source.chirpMult);
    QCOMPARE(dest.downMixSideband, source.downMixSideband);
    QVERIFY(dest.clocks.isEmpty());
}

void LoadoutManagerTest::testFtmwPresetCrud()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout lo;
    lo.name = u"CrudTest"_s;
    lo.hardwareMap = {{u"FtmwDigitizer.main"_s, u"VirtualDigitizer"_s}};

    std::unique_ptr<LoadoutManager> lm(makeLm());
    lm->putLoadout(lo);

    // put and get
    const FtmwPreset p1 = makeFtmwPreset(u"FtmwDigitizer.main"_s);
    QVERIFY(lm->putFtmwPreset(u"CrudTest"_s, u"Alpha"_s, p1));
    QVERIFY(lm->ftmwPresetExists(u"CrudTest"_s, u"Alpha"_s));

    auto got = lm->getFtmwPreset(u"CrudTest"_s, u"Alpha"_s);
    QVERIFY(got.has_value());
    verifyPresetEqual(*got, p1);

    // put __LastUsed__
    QVERIFY(lm->putFtmwPreset(u"CrudTest"_s, lastUsedFtmwPresetName.toString(), p1));
    QVERIFY(lm->ftmwPresetExists(u"CrudTest"_s, lastUsedFtmwPresetName.toString()));

    // ftmwPresetNames excludes __LastUsed__ by default
    QStringList names = lm->ftmwPresetNames(u"CrudTest"_s);
    QVERIFY(names.contains(u"Alpha"_s));
    QVERIFY(!names.contains(lastUsedFtmwPresetName.toString()));

    // includeLastUsed = true includes it
    QStringList allNames = lm->ftmwPresetNames(u"CrudTest"_s, true);
    QVERIFY(allNames.contains(lastUsedFtmwPresetName.toString()));

    // second put emits changed, not added
    QSignalSpy changedSpy(lm.get(), &LoadoutManager::ftmwPresetChanged);
    QSignalSpy addedSpy(lm.get(),   &LoadoutManager::ftmwPresetAdded);
    lm->putFtmwPreset(u"CrudTest"_s, u"Alpha"_s, p1);
    QCOMPARE(changedSpy.count(), 1);
    QCOMPARE(addedSpy.count(), 0);

    // remove
    QSignalSpy removedSpy(lm.get(), &LoadoutManager::ftmwPresetRemoved);
    QVERIFY(lm->removeFtmwPreset(u"CrudTest"_s, u"Alpha"_s));
    QVERIFY(!lm->ftmwPresetExists(u"CrudTest"_s, u"Alpha"_s));
    QCOMPARE(removedSpy.count(), 1);

    // persistence
    lm->putFtmwPreset(u"CrudTest"_s, u"Beta"_s, p1);
    lm.reset();

    std::unique_ptr<LoadoutManager> lm2(makeLm());
    QVERIFY(lm2->ftmwPresetExists(u"CrudTest"_s, u"Beta"_s));
    QVERIFY(!lm2->ftmwPresetExists(u"CrudTest"_s, u"Alpha"_s));
    verifyPresetEqual(*lm2->getFtmwPreset(u"CrudTest"_s, u"Beta"_s), p1);
}

void LoadoutManagerTest::testCurrentFtmwPresetPointers()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout lo;
    lo.name = u"PointerTest"_s;
    lo.hardwareMap = {{u"FtmwDigitizer.main"_s, u"VirtualDigitizer"_s}};

    std::unique_ptr<LoadoutManager> lm(makeLm());
    lm->putLoadout(lo);

    const FtmwPreset p = makeFtmwPreset(u"FtmwDigitizer.main"_s);
    lm->putFtmwPreset(u"PointerTest"_s, u"A"_s, p);
    lm->putFtmwPreset(u"PointerTest"_s, u"B"_s, p);

    // set and get current
    QVERIFY(lm->setCurrentFtmwPresetName(u"PointerTest"_s, u"A"_s));
    QCOMPARE(lm->currentFtmwPresetName(u"PointerTest"_s), u"A"_s);

    // currentFtmwPreset() resolves to the current preset
    auto resolved = lm->currentFtmwPreset(u"PointerTest"_s);
    QVERIFY(resolved.has_value());

    // active preset cannot be removed
    QVERIFY(!lm->removeFtmwPreset(u"PointerTest"_s, u"A"_s));
    QVERIFY(lm->ftmwPresetExists(u"PointerTest"_s, u"A"_s));

    // non-active preset can be removed
    QVERIFY(lm->removeFtmwPreset(u"PointerTest"_s, u"B"_s));
    QVERIFY(!lm->ftmwPresetExists(u"PointerTest"_s, u"B"_s));
    QCOMPARE(lm->currentFtmwPresetName(u"PointerTest"_s), u"A"_s);

    // currentFtmwPreset() returns nullopt when current is empty
    lm->setCurrentFtmwPresetName(u"PointerTest"_s, {});
    QVERIFY(!lm->currentFtmwPreset(u"PointerTest"_s).has_value());

    // pointer persistence
    lm->putFtmwPreset(u"PointerTest"_s, u"C"_s, p);
    lm->setCurrentFtmwPresetName(u"PointerTest"_s, u"C"_s);
    lm.reset();

    std::unique_ptr<LoadoutManager> lm2(makeLm());
    QCOMPARE(lm2->currentFtmwPresetName(u"PointerTest"_s), u"C"_s);
}

void LoadoutManagerTest::testRenameFtmwPresetRewritesPointers()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout lo;
    lo.name = u"RenameTest"_s;
    lo.hardwareMap = {{u"FtmwDigitizer.main"_s, u"VirtualDigitizer"_s}};

    std::unique_ptr<LoadoutManager> lm(makeLm());
    lm->putLoadout(lo);

    const FtmwPreset p = makeFtmwPreset(u"FtmwDigitizer.main"_s);
    lm->putFtmwPreset(u"RenameTest"_s, u"OldName"_s, p);
    lm->setCurrentFtmwPresetName(u"RenameTest"_s, u"OldName"_s);

    // successful rename
    QVERIFY(lm->renameFtmwPreset(u"RenameTest"_s, u"OldName"_s, u"NewName"_s));
    QVERIFY(!lm->ftmwPresetExists(u"RenameTest"_s, u"OldName"_s));
    QVERIFY(lm->ftmwPresetExists(u"RenameTest"_s, u"NewName"_s));
    QCOMPARE(lm->currentFtmwPresetName(u"RenameTest"_s), u"NewName"_s);

    // cannot rename __LastUsed__
    lm->putFtmwPreset(u"RenameTest"_s, lastUsedFtmwPresetName.toString(), p);
    QVERIFY(!lm->renameFtmwPreset(u"RenameTest"_s,
                                   lastUsedFtmwPresetName.toString(), u"SomeName"_s));

    // cannot rename to __LastUsed__
    QVERIFY(!lm->renameFtmwPreset(u"RenameTest"_s, u"NewName"_s,
                                   lastUsedFtmwPresetName.toString()));

    // cannot rename to a duplicate
    lm->putFtmwPreset(u"RenameTest"_s, u"Other"_s, p);
    QVERIFY(!lm->renameFtmwPreset(u"RenameTest"_s, u"NewName"_s, u"Other"_s));

    // persistence
    lm.reset();
    std::unique_ptr<LoadoutManager> lm2(makeLm());
    QVERIFY(lm2->ftmwPresetExists(u"RenameTest"_s, u"NewName"_s));
    QVERIFY(!lm2->ftmwPresetExists(u"RenameTest"_s, u"OldName"_s));
    QCOMPARE(lm2->currentFtmwPresetName(u"RenameTest"_s), u"NewName"_s);
}

void LoadoutManagerTest::testRemoveLoadoutCascadesFtmwPresets()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout lo;
    lo.name = u"CascadeTest"_s;
    lo.hardwareMap = {{u"FtmwDigitizer.main"_s, u"VirtualDigitizer"_s}};

    std::unique_ptr<LoadoutManager> lm(makeLm());
    lm->putLoadout(lo);
    lm->putFtmwPreset(u"CascadeTest"_s, u"P1"_s, makeFtmwPreset(u"FtmwDigitizer.main"_s));
    lm->putFtmwPreset(u"CascadeTest"_s, u"P2"_s, makeFtmwPreset(u"FtmwDigitizer.main"_s));
    QCOMPARE(lm->ftmwPresetNames(u"CascadeTest"_s).size(), 2);

    QVERIFY(lm->removeLoadout(u"CascadeTest"_s));
    QVERIFY(!lm->loadoutExists(u"CascadeTest"_s));

    // After reload the loadout and its presets are gone
    lm.reset();
    std::unique_ptr<LoadoutManager> lm2(makeLm());
    QVERIFY(!lm2->loadoutExists(u"CascadeTest"_s));
    QCOMPARE(lm2->ftmwPresetNames(u"CascadeTest"_s).size(), 0);
}

void LoadoutManagerTest::testRoundTripWithLifPresets()
{
    using namespace Qt::StringLiterals;
    const HardwareLoadout original = makeWithLifPresets();

    {
        std::unique_ptr<LoadoutManager> lm(makeLm());
        QVERIFY(lm->putLoadout(original));
    }

    std::unique_ptr<LoadoutManager> lm2(makeLm());
    QVERIFY(lm2->loadoutExists(original.name));

    const auto got = lm2->getLoadout(original.name);
    QVERIFY(got.has_value());
    QCOMPARE(got->name,                original.name);
    QCOMPARE(got->hardwareMap,         original.hardwareMap);
    QCOMPARE(got->currentLifPresetName, original.currentLifPresetName);

    // All three presets present
    QCOMPARE(got->lifPresets.size(), std::size_t(3));
    QVERIFY(got->lifPresets.count(u"Primary"_s));
    QVERIFY(got->lifPresets.count(u"Secondary"_s));
    QVERIFY(got->lifPresets.count(lastUsedLifPresetName.toString()));

    verifyLifPresetEqual(got->lifPresets.at(u"Primary"_s),
                         original.lifPresets.at(u"Primary"_s));
    verifyLifPresetEqual(got->lifPresets.at(u"Secondary"_s),
                         original.lifPresets.at(u"Secondary"_s));
}

void LoadoutManagerTest::testLifPresetCrud()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout lo;
    lo.name = u"LifCrudTest"_s;
    lo.hardwareMap = {{u"LifLaser.default"_s, u"VirtualLifLaser"_s}};

    std::unique_ptr<LoadoutManager> lm(makeLm());
    lm->putLoadout(lo);

    // put and get
    const LifPreset p1 = makeLifPreset(u"LifLaser.default"_s);
    QVERIFY(lm->putLifPreset(u"LifCrudTest"_s, u"Alpha"_s, p1));
    QVERIFY(lm->lifPresetExists(u"LifCrudTest"_s, u"Alpha"_s));

    auto got = lm->getLifPreset(u"LifCrudTest"_s, u"Alpha"_s);
    QVERIFY(got.has_value());
    verifyLifPresetEqual(*got, p1);

    // put __LastUsed__
    QVERIFY(lm->putLifPreset(u"LifCrudTest"_s, lastUsedLifPresetName.toString(), p1));
    QVERIFY(lm->lifPresetExists(u"LifCrudTest"_s, lastUsedLifPresetName.toString()));

    // lifPresetNames excludes __LastUsed__ by default
    QStringList names = lm->lifPresetNames(u"LifCrudTest"_s);
    QVERIFY(names.contains(u"Alpha"_s));
    QVERIFY(!names.contains(lastUsedLifPresetName.toString()));

    // includeLastUsed = true includes it
    QStringList allNames = lm->lifPresetNames(u"LifCrudTest"_s, true);
    QVERIFY(allNames.contains(lastUsedLifPresetName.toString()));

    // second put emits changed, not added
    QSignalSpy changedSpy(lm.get(), &LoadoutManager::lifPresetChanged);
    QSignalSpy addedSpy(lm.get(),   &LoadoutManager::lifPresetAdded);
    lm->putLifPreset(u"LifCrudTest"_s, u"Alpha"_s, p1);
    QCOMPARE(changedSpy.count(), 1);
    QCOMPARE(addedSpy.count(), 0);

    // remove
    QSignalSpy removedSpy(lm.get(), &LoadoutManager::lifPresetRemoved);
    QVERIFY(lm->removeLifPreset(u"LifCrudTest"_s, u"Alpha"_s));
    QVERIFY(!lm->lifPresetExists(u"LifCrudTest"_s, u"Alpha"_s));
    QCOMPARE(removedSpy.count(), 1);

    // persistence
    lm->putLifPreset(u"LifCrudTest"_s, u"Beta"_s, p1);
    lm.reset();

    std::unique_ptr<LoadoutManager> lm2(makeLm());
    QVERIFY(lm2->lifPresetExists(u"LifCrudTest"_s, u"Beta"_s));
    QVERIFY(!lm2->lifPresetExists(u"LifCrudTest"_s, u"Alpha"_s));
    verifyLifPresetEqual(*lm2->getLifPreset(u"LifCrudTest"_s, u"Beta"_s), p1);
}

void LoadoutManagerTest::testCurrentLifPresetPointers()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout lo;
    lo.name = u"LifPointerTest"_s;
    lo.hardwareMap = {{u"LifLaser.default"_s, u"VirtualLifLaser"_s}};

    std::unique_ptr<LoadoutManager> lm(makeLm());
    lm->putLoadout(lo);

    const LifPreset p = makeLifPreset(u"LifLaser.default"_s);
    lm->putLifPreset(u"LifPointerTest"_s, u"A"_s, p);
    lm->putLifPreset(u"LifPointerTest"_s, u"B"_s, p);

    // set and get current
    QVERIFY(lm->setCurrentLifPresetName(u"LifPointerTest"_s, u"A"_s));
    QCOMPARE(lm->currentLifPresetName(u"LifPointerTest"_s), u"A"_s);

    // currentLifPreset() resolves to the current preset
    auto resolved = lm->currentLifPreset(u"LifPointerTest"_s);
    QVERIFY(resolved.has_value());

    // active preset cannot be removed
    QVERIFY(!lm->removeLifPreset(u"LifPointerTest"_s, u"A"_s));
    QVERIFY(lm->lifPresetExists(u"LifPointerTest"_s, u"A"_s));

    // non-active preset can be removed
    QVERIFY(lm->removeLifPreset(u"LifPointerTest"_s, u"B"_s));
    QVERIFY(!lm->lifPresetExists(u"LifPointerTest"_s, u"B"_s));
    QCOMPARE(lm->currentLifPresetName(u"LifPointerTest"_s), u"A"_s);

    // currentLifPreset() returns nullopt when current is empty
    lm->setCurrentLifPresetName(u"LifPointerTest"_s, {});
    QVERIFY(!lm->currentLifPreset(u"LifPointerTest"_s).has_value());

    // pointer persistence
    lm->putLifPreset(u"LifPointerTest"_s, u"C"_s, p);
    lm->setCurrentLifPresetName(u"LifPointerTest"_s, u"C"_s);
    lm.reset();

    std::unique_ptr<LoadoutManager> lm2(makeLm());
    QCOMPARE(lm2->currentLifPresetName(u"LifPointerTest"_s), u"C"_s);
}

void LoadoutManagerTest::testRenameLifPresetRewritesPointers()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout lo;
    lo.name = u"LifRenameTest"_s;
    lo.hardwareMap = {{u"LifLaser.default"_s, u"VirtualLifLaser"_s}};

    std::unique_ptr<LoadoutManager> lm(makeLm());
    lm->putLoadout(lo);

    const LifPreset p = makeLifPreset(u"LifLaser.default"_s);
    lm->putLifPreset(u"LifRenameTest"_s, u"OldName"_s, p);
    lm->setCurrentLifPresetName(u"LifRenameTest"_s, u"OldName"_s);

    // successful rename
    QVERIFY(lm->renameLifPreset(u"LifRenameTest"_s, u"OldName"_s, u"NewName"_s));
    QVERIFY(!lm->lifPresetExists(u"LifRenameTest"_s, u"OldName"_s));
    QVERIFY(lm->lifPresetExists(u"LifRenameTest"_s, u"NewName"_s));
    QCOMPARE(lm->currentLifPresetName(u"LifRenameTest"_s), u"NewName"_s);

    // cannot rename __LastUsed__
    lm->putLifPreset(u"LifRenameTest"_s, lastUsedLifPresetName.toString(), p);
    QVERIFY(!lm->renameLifPreset(u"LifRenameTest"_s,
                                  lastUsedLifPresetName.toString(), u"SomeName"_s));

    // cannot rename to __LastUsed__
    QVERIFY(!lm->renameLifPreset(u"LifRenameTest"_s, u"NewName"_s,
                                  lastUsedLifPresetName.toString()));

    // cannot rename to a duplicate
    lm->putLifPreset(u"LifRenameTest"_s, u"Other"_s, p);
    QVERIFY(!lm->renameLifPreset(u"LifRenameTest"_s, u"NewName"_s, u"Other"_s));

    // persistence
    lm.reset();
    std::unique_ptr<LoadoutManager> lm2(makeLm());
    QVERIFY(lm2->lifPresetExists(u"LifRenameTest"_s, u"NewName"_s));
    QVERIFY(!lm2->lifPresetExists(u"LifRenameTest"_s, u"OldName"_s));
    QCOMPARE(lm2->currentLifPresetName(u"LifRenameTest"_s), u"NewName"_s);
}

void LoadoutManagerTest::testRemoveLoadoutCascadesLifPresets()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout lo;
    lo.name = u"LifCascadeTest"_s;
    lo.hardwareMap = {{u"LifLaser.default"_s, u"VirtualLifLaser"_s}};

    std::unique_ptr<LoadoutManager> lm(makeLm());
    lm->putLoadout(lo);
    lm->putLifPreset(u"LifCascadeTest"_s, u"P1"_s, makeLifPreset(u"LifLaser.default"_s));
    lm->putLifPreset(u"LifCascadeTest"_s, u"P2"_s, makeLifPreset(u"LifLaser.default"_s));
    QCOMPARE(lm->lifPresetNames(u"LifCascadeTest"_s).size(), 2);

    QVERIFY(lm->removeLoadout(u"LifCascadeTest"_s));
    QVERIFY(!lm->loadoutExists(u"LifCascadeTest"_s));

    // After reload the loadout and its presets are gone
    lm.reset();
    std::unique_ptr<LoadoutManager> lm2(makeLm());
    QVERIFY(!lm2->loadoutExists(u"LifCascadeTest"_s));
    QCOMPARE(lm2->lifPresetNames(u"LifCascadeTest"_s).size(), 0);
}

void LoadoutManagerTest::testLifConversionWiringArrayRoundTrip()
{
    using namespace Qt::StringLiterals;
    using namespace BC::LifConv;

    LifConversionSnapshot snap;
    snap.laserKey = u"LifLaser.default"_s;

    StageWiring doubler;
    doubler.stageKey = u"doubler"_s;
    doubler.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    doubler.isFinal = false;
    snap.wiring.push_back(doubler);

    StageWiring sfg;
    sfg.stageKey = u"sfg"_s;
    sfg.inputs = {InputRef{RefType::Stage, u"doubler"_s, 0.0},
                 InputRef{RefType::Fixed, {}, 123.456}};
    sfg.isFinal = true;
    snap.wiring.push_back(sfg);

    const auto scalars = lifConversionScalarsMap(snap);
    const auto wiring  = lifConversionWiringArray(snap);
    const LifConversionSnapshot back = lifConversionSnapshotFromMaps(scalars, wiring);

    QCOMPARE(back.laserKey, snap.laserKey);
    QCOMPARE(back.wiring.size(), snap.wiring.size());
    for (std::size_t i = 0; i < snap.wiring.size(); ++i) {
        QCOMPARE(back.wiring[i].stageKey, snap.wiring[i].stageKey);
        QCOMPARE(back.wiring[i].isFinal,  snap.wiring[i].isFinal);
        QCOMPARE(back.wiring[i].inputs.size(), snap.wiring[i].inputs.size());
        for (std::size_t j = 0; j < snap.wiring[i].inputs.size(); ++j) {
            QCOMPARE(back.wiring[i].inputs[j].type,     snap.wiring[i].inputs[j].type);
            QCOMPARE(back.wiring[i].inputs[j].stageKey, snap.wiring[i].inputs[j].stageKey);
            QCOMPARE(back.wiring[i].inputs[j].fixedCm1, snap.wiring[i].inputs[j].fixedCm1);
        }
    }
}

void LoadoutManagerTest::testClearLifPresets()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout lo = makeWithLifPresets();
    lo.name = u"LifClearTest"_s;

    std::unique_ptr<LoadoutManager> lm(makeLm());
    QVERIFY(lm->putLoadout(lo));
    QCOMPARE(lm->lifPresetNames(u"LifClearTest"_s, true).size(), 3);
    QCOMPARE(lm->currentLifPresetName(u"LifClearTest"_s), u"Secondary"_s);

    QSignalSpy changedSpy(lm.get(), &LoadoutManager::loadoutChanged);
    QVERIFY(lm->clearLifPresets(u"LifClearTest"_s));
    QCOMPARE(changedSpy.count(), 1);

    // In-memory state is cleared, hardware map is untouched
    QVERIFY(lm->lifPresetNames(u"LifClearTest"_s, true).isEmpty());
    QVERIFY(lm->currentLifPresetName(u"LifClearTest"_s).isEmpty());
    const auto got = lm->getLoadout(u"LifClearTest"_s);
    QVERIFY(got.has_value());
    QCOMPARE(got->hardwareMap, lo.hardwareMap);

    // Clearing an unknown loadout fails
    QVERIFY(!lm->clearLifPresets(u"NoSuchLoadout"_s));

    // Persistence: the on-disk subtree is actually purged, not just the index
    lm.reset();
    std::unique_ptr<LoadoutManager> lm2(makeLm());
    QVERIFY(lm2->lifPresetNames(u"LifClearTest"_s, true).isEmpty());
    QVERIFY(lm2->currentLifPresetName(u"LifClearTest"_s).isEmpty());
    QVERIFY(!lm2->lifPresetExists(u"LifClearTest"_s, u"Primary"_s));
    QCOMPARE(lm2->getLoadout(u"LifClearTest"_s)->hardwareMap, lo.hardwareMap);
}

void LoadoutManagerTest::testPutLoadoutPreservesLifPresetsAcrossResave()
{
    // Regression test for the loadout Save handlers wiping lifPresets: as
    // long as the caller carries `lifPresets`/`currentLifPresetName` forward
    // from the existing loadout into the struct passed to putLoadout(),
    // presets survive a re-save exactly like ftmwPresets already do.
    using namespace Qt::StringLiterals;

    const HardwareLoadout original = makeWithLifPresets();
    std::unique_ptr<LoadoutManager> lm(makeLm());
    QVERIFY(lm->putLoadout(original));

    const auto existing = lm->getLoadout(original.name);
    QVERIFY(existing.has_value());

    // Simulate a hardware-config re-save that rebuilds a fresh struct (as
    // onLoadoutSave() does) but correctly forwards the LIF preset family.
    HardwareLoadout resaved;
    resaved.name = original.name;
    resaved.hardwareMap = existing->hardwareMap;
    resaved.lifPresets = existing->lifPresets;
    resaved.currentLifPresetName = existing->currentLifPresetName;
    resaved.lastModified = QDateTime::currentDateTimeUtc();

    QVERIFY(lm->putLoadout(resaved));

    const auto got = lm->getLoadout(original.name);
    QVERIFY(got.has_value());
    QCOMPARE(got->currentLifPresetName, original.currentLifPresetName);
    QCOMPARE(got->lifPresets.size(), original.lifPresets.size());
    verifyLifPresetEqual(got->lifPresets.at(u"Primary"_s), original.lifPresets.at(u"Primary"_s));
    verifyLifPresetEqual(got->lifPresets.at(u"Secondary"_s), original.lifPresets.at(u"Secondary"_s));

    // Persistence after reload
    lm.reset();
    std::unique_ptr<LoadoutManager> lm2(makeLm());
    const auto got2 = lm2->getLoadout(original.name);
    QVERIFY(got2.has_value());
    QCOMPARE(got2->lifPresets.size(), original.lifPresets.size());
    QCOMPARE(got2->currentLifPresetName, original.currentLifPresetName);
}

void LoadoutManagerTest::testHardwareIdentityRoundTrip()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout original = makeWithPresets();
    original.name = u"IdentityRoundTrip"_s;
    original.hardwareIdentity = {
        {u"Clock.ref"_s,           u"idclock"_s},
        {u"AWG.main"_s,            u"idawg"_s},
        {u"FtmwDigitizer.main"_s,  u"iddigi"_s},
    };

    {
        std::unique_ptr<LoadoutManager> lm(makeLm());
        QVERIFY(lm->putLoadout(original));
    }

    std::unique_ptr<LoadoutManager> lm2(makeLm());
    const auto got = lm2->getLoadout(original.name);
    QVERIFY(got.has_value());
    QCOMPARE(got->hardwareMap,      original.hardwareMap);
    QCOMPARE(got->hardwareIdentity, original.hardwareIdentity);
}

void LoadoutManagerTest::testHardwareIdentityMigration()
{
    using namespace Qt::StringLiterals;

    // A loadout written before identity tracking has a populated hardware map
    // but no identity fields on any record. On reload the identity map must
    // come back empty (a wildcard), with no synthesized values.
    HardwareLoadout original = makeHardwareOnly();
    original.name = u"IdentityMigration"_s;
    QVERIFY(original.hardwareIdentity.empty());
    QVERIFY(!original.hardwareMap.empty());

    {
        std::unique_ptr<LoadoutManager> lm(makeLm());
        QVERIFY(lm->putLoadout(original));
    }

    std::unique_ptr<LoadoutManager> lm2(makeLm());
    const auto got = lm2->getLoadout(original.name);
    QVERIFY(got.has_value());
    QCOMPARE(got->hardwareMap, original.hardwareMap);
    QVERIFY(got->hardwareIdentity.empty());
}

void LoadoutManagerTest::testFtmwPresetReferencesHardware()
{
    using namespace Qt::StringLiterals;

    const FtmwPreset preset = makeFtmwPreset(u"FtmwDigitizer.main"_s);

    // Digitizer hwKey and each referenced clock hwKey match.
    QVERIFY(ftmwPresetReferencesHardware(preset, u"FtmwDigitizer.main"_s));
    QVERIFY(ftmwPresetReferencesHardware(preset, u"Clock.ref"_s));
    QVERIFY(ftmwPresetReferencesHardware(preset, u"Clock.lo"_s));

    // An unrelated hwKey does not.
    QVERIFY(!ftmwPresetReferencesHardware(preset, u"Clock.unrelated"_s));
}

void LoadoutManagerTest::testLifPresetReferencesHardware()
{
    using namespace Qt::StringLiterals;

    const LifPreset preset = makeLifPreset(u"LifLaser.default"_s);

    // A wired stageKey and the captured laser match.
    QVERIFY(lifPresetReferencesHardware(preset, u"LifFreqConversionStage.doubler"_s));
    QVERIFY(lifPresetReferencesHardware(preset, u"LifFreqConversionStage.tripler"_s));
    QVERIFY(lifPresetReferencesHardware(preset, u"LifLaser.default"_s));

    // An unrelated hwKey does not.
    QVERIFY(!lifPresetReferencesHardware(preset, u"LifLaser.other"_s));
}

QTEST_GUILESS_MAIN(LoadoutManagerTest)
#include "tst_loadoutmanagertest.moc"
