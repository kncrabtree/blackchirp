#include <QtTest>
#include <QCoreApplication>
#include <QSettings>
#include <QTemporaryDir>

#include <src/data/loadout/loadoutmanager.h>
#include <src/data/loadout/hardwareloadout.h>

using namespace BC::Loadout;

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

    void testRoundTripFull();
    void testRoundTripNoFtmw();
    void testRemoveLoadout();
    void testCurrentDefaultPersistence();
    void testClockArrayRoundTrip();
    void testDigitizerChannelArrayRoundTrip();
    void testLoadoutsMatchingHwKey();
    void testCopyClocksMatching();
    void testCopyRfScalars();

private:
    // Creates a fresh isolated LoadoutManager (bypasses the singleton).
    LoadoutManager *makeLm() const;

    static HardwareLoadout makeFull();
    static void verifyClocksEqual(const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &a,
                                  const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &b);

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
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, d_tempDir->path());
}

void LoadoutManagerTest::cleanupTestCase()
{
    delete d_tempDir;
    d_tempDir = nullptr;
}

void LoadoutManagerTest::init()
{
    // Wipe all settings between tests so each starts from a clean state.
    QSettings s(QSettings::IniFormat, QSettings::UserScope, s_org, s_app);
    s.clear();
    s.sync();
}

LoadoutManager *LoadoutManagerTest::makeLm() const
{
    return new LoadoutManager(s_org, s_app);
}

HardwareLoadout LoadoutManagerTest::makeFull()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout lo;
    lo.name = u"Alpha"_s;
    lo.hardwareMap = {
        {u"Clock.ref"_s,       u"VirtualClock"_s},
        {u"AWG.main"_s,        u"VirtualAwg"_s},
        {u"FtmwScope.main"_s,  u"VirtualScope"_s},
    };

    FtmwSnapshot snap;
    snap.digiHwKey = u"FtmwScope.main"_s;

    // RF scalars
    snap.rfConfig.commonUpDownLO  = true;
    snap.rfConfig.awgMult         = 2.5;
    snap.rfConfig.upMixSideband   = RfConfig::LowerSideband;
    snap.rfConfig.chirpMult       = 4.0;
    snap.rfConfig.downMixSideband = RfConfig::UpperSideband;

    // Two clocks
    {
        RfConfig::ClockFreq cf;
        cf.hwKey          = u"Clock.ref"_s;
        cf.output         = 1;
        cf.op             = RfConfig::Multiply;
        cf.factor         = 10.0;
        cf.desiredFreqMHz = 100.0;
        snap.rfConfig.clocks.insert(RfConfig::AwgRef, cf);
    }
    {
        RfConfig::ClockFreq cf;
        cf.hwKey          = u"Clock.lo"_s;
        cf.output         = 0;
        cf.op             = RfConfig::Divide;
        cf.factor         = 2.0;
        cf.desiredFreqMHz = 8500.0;
        snap.rfConfig.clocks.insert(RfConfig::UpLO, cf);
    }

    // ChirpConfig
    snap.chirpConfig.setAwgSampleRate(4e9);
    snap.chirpConfig.setNumChirps(2);
    snap.chirpConfig.setChirpInterval(500.0);
    snap.chirpConfig.addSegment(7000.0, 8000.0, 5.0, 0);
    snap.chirpConfig.addSegment(7000.0, 8000.0, 5.0, 1);

    QVector<MarkerChannel> markers;
    MarkerChannel mc;
    mc.name      = u"ProtectionGate"_s;
    mc.role      = MarkerRole::Protection;
    mc.timingMode = MarkerChannel::ChirpRelative;
    mc.startTime = -0.25;
    mc.endTime   = 0.25;
    mc.enabled   = true;
    markers.push_back(mc);
    snap.chirpConfig.setMarkerChannels(markers);

    // Digitizer
    snap.digitizer = FtmwDigitizerConfig(snap.digiHwKey);
    snap.digitizer.d_triggerChannel   = 2;
    snap.digitizer.d_triggerSlope     = DigitizerConfig::FallingEdge;
    snap.digitizer.d_triggerDelayUSec = 0.1;
    snap.digitizer.d_triggerLevel     = 0.5;
    snap.digitizer.d_sampleRate       = 4e9;
    snap.digitizer.d_recordLength     = 1024;
    snap.digitizer.d_bytesPerPoint    = 2;
    snap.digitizer.d_byteOrder        = DigitizerConfig::LittleEndian;
    snap.digitizer.d_blockAverage     = true;
    snap.digitizer.d_numAverages      = 100;
    snap.digitizer.d_multiRecord      = false;
    snap.digitizer.d_numRecords       = 1;
    snap.digitizer.d_fidChannel       = 3;

    DigitizerConfig::AnalogChannel ach;
    ach.enabled   = true;
    ach.fullScale = 0.1;
    ach.offset    = 0.02;
    snap.digitizer.d_analogChannels[1] = ach;

    DigitizerConfig::DigitalChannel dch;
    dch.enabled = true;
    dch.input   = false;
    dch.role    = 5;
    snap.digitizer.d_digitalChannels[0] = dch;

    lo.ftmw = std::move(snap);
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

// ── test cases ────────────────────────────────────────────────────────────────

void LoadoutManagerTest::testRoundTripFull()
{
    const HardwareLoadout original = makeFull();

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

    QVERIFY(got->ftmw.has_value());
    const auto &os = *original.ftmw;
    const auto &gs = *got->ftmw;

    QCOMPARE(gs.digiHwKey, os.digiHwKey);

    // RF scalars
    QCOMPARE(gs.rfConfig.commonUpDownLO,  os.rfConfig.commonUpDownLO);
    QCOMPARE(gs.rfConfig.awgMult,         os.rfConfig.awgMult);
    QCOMPARE(gs.rfConfig.upMixSideband,   os.rfConfig.upMixSideband);
    QCOMPARE(gs.rfConfig.chirpMult,       os.rfConfig.chirpMult);
    QCOMPARE(gs.rfConfig.downMixSideband, os.rfConfig.downMixSideband);

    // Clocks
    verifyClocksEqual(gs.rfConfig.clocks, os.rfConfig.clocks);

    // ChirpConfig
    QCOMPARE(gs.chirpConfig.numChirps(),     os.chirpConfig.numChirps());
    QCOMPARE(gs.chirpConfig.chirpInterval(), os.chirpConfig.chirpInterval());
    const auto &gcl = gs.chirpConfig.chirpList();
    const auto &ocl = os.chirpConfig.chirpList();
    QCOMPARE(gcl.size(), ocl.size());
    for (int ci = 0; ci < ocl.size(); ++ci) {
        QCOMPARE(gcl[ci].size(), ocl[ci].size());
        for (int si = 0; si < ocl[ci].size(); ++si) {
            QCOMPARE(gcl[ci][si].startFreqMHz, ocl[ci][si].startFreqMHz);
            QCOMPARE(gcl[ci][si].endFreqMHz,   ocl[ci][si].endFreqMHz);
            QCOMPARE(gcl[ci][si].durationUs,   ocl[ci][si].durationUs);
            QCOMPARE(gcl[ci][si].alphaUs,      ocl[ci][si].alphaUs);
            QCOMPARE(gcl[ci][si].empty,        ocl[ci][si].empty);
        }
    }
    QCOMPARE(gs.chirpConfig.markerChannels().size(),
             os.chirpConfig.markerChannels().size());
    const auto &gm = gs.chirpConfig.markerChannels().at(0);
    const auto &om = os.chirpConfig.markerChannels().at(0);
    QCOMPARE(gm.name,       om.name);
    QCOMPARE(gm.role,       om.role);
    QCOMPARE(gm.timingMode, om.timingMode);
    QCOMPARE(gm.startTime,  om.startTime);
    QCOMPARE(gm.endTime,    om.endTime);
    QCOMPARE(gm.enabled,    om.enabled);

    // Digitizer scalars
    const auto &gd = gs.digitizer;
    const auto &od = os.digitizer;
    QCOMPARE(gd.d_triggerChannel,   od.d_triggerChannel);
    QCOMPARE(gd.d_triggerSlope,     od.d_triggerSlope);
    QCOMPARE(gd.d_triggerDelayUSec, od.d_triggerDelayUSec);
    QCOMPARE(gd.d_triggerLevel,     od.d_triggerLevel);
    QCOMPARE(gd.d_sampleRate,       od.d_sampleRate);
    QCOMPARE(gd.d_recordLength,     od.d_recordLength);
    QCOMPARE(gd.d_bytesPerPoint,    od.d_bytesPerPoint);
    QCOMPARE(gd.d_byteOrder,        od.d_byteOrder);
    QCOMPARE(gd.d_blockAverage,     od.d_blockAverage);
    QCOMPARE(gd.d_numAverages,      od.d_numAverages);
    QCOMPARE(gd.d_multiRecord,      od.d_multiRecord);
    QCOMPARE(gd.d_numRecords,       od.d_numRecords);
    QCOMPARE(gd.d_fidChannel,       od.d_fidChannel);

    // Analog channel
    QCOMPARE(gd.d_analogChannels.size(), od.d_analogChannels.size());
    const auto &gach = gd.d_analogChannels.at(1);
    const auto &oach = od.d_analogChannels.at(1);
    QCOMPARE(gach.enabled,   oach.enabled);
    QCOMPARE(gach.fullScale, oach.fullScale);
    QCOMPARE(gach.offset,    oach.offset);

    // Digital channel
    QCOMPARE(gd.d_digitalChannels.size(), od.d_digitalChannels.size());
    const auto &gdch = gd.d_digitalChannels.at(0);
    const auto &odch = od.d_digitalChannels.at(0);
    QCOMPARE(gdch.enabled, odch.enabled);
    QCOMPARE(gdch.input,   odch.input);
    QCOMPARE(gdch.role,    odch.role);
}

void LoadoutManagerTest::testRoundTripNoFtmw()
{
    using namespace Qt::StringLiterals;

    HardwareLoadout lo;
    lo.name = u"NoFtmw"_s;
    lo.hardwareMap = {{u"AWG.main"_s, u"VirtualAwg"_s}};
    // ftmw intentionally absent

    {
        std::unique_ptr<LoadoutManager> lm(makeLm());
        QVERIFY(lm->putLoadout(lo));
    }

    std::unique_ptr<LoadoutManager> lm2(makeLm());
    const auto got = lm2->getLoadout(lo.name);
    QVERIFY(got.has_value());
    QCOMPARE(got->name,        lo.name);
    QCOMPARE(got->hardwareMap, lo.hardwareMap);
    QVERIFY(!got->ftmw.has_value());
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
        // current should have been reassigned away from the removed loadout
        QVERIFY(lm->currentLoadoutName() != u"Alpha"_s);
    }

    // Verify the removal is persisted
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

    // Three clocks, all six fields distinct
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

    FtmwDigitizerConfig cfg(u"FtmwScope.test"_s);
    cfg.d_analogChannels[0]  = {true,  0.05, -0.01};
    cfg.d_analogChannels[1]  = {false, 0.5,   0.0};
    cfg.d_digitalChannels[0] = {true,  true,  2};
    cfg.d_digitalChannels[1] = {true,  false, 7};
    cfg.d_fidChannel = 1;

    const auto analog   = digitizerAnalogArray(cfg);
    const auto digital  = digitizerDigitalArray(cfg);
    const auto scalars  = digitizerScalarsMap(cfg);
    const FtmwDigitizerConfig back = ftmwDigitizerFromMaps(
        u"FtmwScope.test"_s, scalars, analog, digital);

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

    const QStringList noMatches = lm->loadoutsMatchingHwKey(u"FtmwScope.main"_s);
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

    // dest starts with a pre-existing clock whose hwKey is not in allowedHwKeys
    RfConfigSnapshot dest;
    {
        RfConfig::ClockFreq cf;
        cf.hwKey = u"Clock.other"_s;
        cf.desiredFreqMHz = 50.0;
        dest.clocks.insert(RfConfig::DigRef, cf);
    }

    const std::set<QString> allowed = {u"Clock.awg-ref"_s};
    copyClocksMatching(source, dest, allowed);

    // AwgRef was in allowed → copied
    QVERIFY(dest.clocks.contains(RfConfig::AwgRef));
    QCOMPARE(dest.clocks[RfConfig::AwgRef].hwKey, u"Clock.awg-ref"_s);

    // UpLO was not in allowed → not copied
    QVERIFY(!dest.clocks.contains(RfConfig::UpLO));

    // Pre-existing DigRef (hwKey="Clock.other") must be preserved
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

    // Clocks must not have been touched
    QVERIFY(dest.clocks.isEmpty());
}

QTEST_GUILESS_MAIN(LoadoutManagerTest)
#include "tst_loadoutmanagertest.moc"
