#include <data/experiment/ftmwconfig.h>

#include <cstring>
#include <math.h>

#include <QFile>
#include <QtEndian>

#include <data/analysis/waveformparser.h>
#include <data/storage/blackchirpcsv.h>
#include <data/storage/waveformbuffer.h>
#include <data/storage/fidpeakupstorage.h>

FtmwConfig::FtmwConfig(const QString& scopeHwKey) : HeaderStorage(BC::Store::FTMW::key)
{
    ps_scopeConfig = std::make_shared<FtmwDigitizerConfig>(scopeHwKey);
}

FtmwConfig::~FtmwConfig()
{

}

quint64 FtmwConfig::shotIncrement() const
{
    quint64 increment = 1;
    if(ps_scopeConfig->d_blockAverage)
        increment *= ps_scopeConfig->d_numAverages;

    return increment;
}

FidList FtmwConfig::parseWaveform(const QByteArray b) const
{
    int np = ps_scopeConfig->d_recordLength;
    int numRec = ps_scopeConfig->d_numRecords;

    QVector<qint64> buf(np * numRec);
    BC::Analysis::parseWaveform(b.constData(), buf.data(),
                                np, numRec,
                                ps_scopeConfig->d_bytesPerPoint,
                                ps_scopeConfig->d_byteOrder,
                                shotIncrement(), bitShift());

    FidList out;
    out.reserve(numRec);
    for(int j = 0; j < numRec; ++j)
    {
        QVector<qint64> d(np);
        memcpy(d.data(), buf.constData() + j*np, np*sizeof(qint64));

        Fid f = d_fidTemplate;
        f.setData(d);
        f.setShots(shotIncrement());
        out.append(f);
    }

    return out;
}

FidList FtmwConfig::parseBatchFids(const std::vector<WaveformEntry> &entries) const
{
    int np = ps_scopeConfig->d_recordLength;
    int numRec = ps_scopeConfig->d_numRecords;

    QVector<qint64> buf(np * numRec, 0);
    BC::Analysis::parseBatchParallel(entries, buf.data(), np, numRec,
                                     ps_scopeConfig->d_bytesPerPoint,
                                     ps_scopeConfig->d_byteOrder,
                                     shotIncrement(), bitShift());

    quint64 totalShots = 0;
    for(const auto &e : entries)
        totalShots += e.shotCount;

    FidList out;
    out.reserve(numRec);
    for(int j = 0; j < numRec; ++j)
    {
        QVector<qint64> d(np);
        memcpy(d.data(), buf.constData() + j*np, np*sizeof(qint64));

        Fid f = d_fidTemplate;
        f.setData(d);
        f.setShots(totalShots);
        out.append(f);
    }

    return out;
}

bool FtmwConfig::addBatchFids(const std::vector<WaveformEntry> &entries)
{
    d_errorString.clear();

    FidList combined = parseBatchFids(entries);

    if(d_chirpScoringEnabled || d_phaseCorrectionEnabled)
    {
        if(!preprocessChirp(combined))
            return true;
    }

#ifdef BC_CUDA
    // CUDA path expects raw bytes; batch path not supported, fall through to storage
    return p_fidStorage->addFids(combined, d_currentShift);
#else
    return p_fidStorage->addFids(combined, d_currentShift);
#endif
}

double FtmwConfig::ftMinMHz() const
{
    double sign = 1.0;
    if(d_rfConfig.d_downMixSideband == RfConfig::LowerSideband)
        sign = -1.0;
    double lo = d_rfConfig.clockFrequency(RfConfig::DownLO);
    double lastFreq = lo + sign*ftNyquistMHz();
    return qMin(lo,lastFreq);
}

double FtmwConfig::ftMaxMHz() const
{
    double sign = 1.0;
    if(d_rfConfig.d_downMixSideband == RfConfig::LowerSideband)
        sign = -1.0;
    double lo = d_rfConfig.clockFrequency(RfConfig::DownLO);
    double lastFreq = lo + sign*ftNyquistMHz();
    return qMax(lo,lastFreq);
}

double FtmwConfig::ftNyquistMHz() const
{
    return ps_scopeConfig->d_sampleRate/(1e6*2.0);
}

double FtmwConfig::fidDurationUs() const
{
    double sr = ps_scopeConfig->d_sampleRate;
    double rl = static_cast<double>(ps_scopeConfig->d_recordLength);

    return rl/sr*1e6;
}

QPair<int, int> FtmwConfig::chirpRange() const
{
    //compute chirp duration in samples (only use first chirp if there are multiple)
    auto dur = d_rfConfig.d_chirpConfig.chirpDurationUs(0);
    if(ps_scopeConfig->d_sampleRate <= 0.0)
        return {0,0};

    int samples = dur*1e-6*ps_scopeConfig->d_sampleRate;

    //compute where the chirp starts in the digitizer's time frame.
    //if a Trigger marker is active, use its start time (negative = before chirp start,
    //so -startTime gives the interval from trigger fire to chirp start).
    //otherwise fall back to leadTimeUs(), assuming the digitizer is triggered at the
    //beginning of the waveform (protection pulse leading edge).
    double startUs = 0.0;
    if(d_chirpOffsetUs < 0.0)
    {
        auto cc = d_rfConfig.d_chirpConfig;
        auto *trig = cc.findEnabledMarkerByRole(MarkerRole::Trigger);
        if(trig)
            startUs = -trig->startTime - ps_scopeConfig->d_triggerDelayUSec;
        else
            startUs = cc.leadTimeUs() - ps_scopeConfig->d_triggerDelayUSec;
    }
    else
        startUs = d_chirpOffsetUs;

    int startSample = startUs*1e-6*ps_scopeConfig->d_sampleRate;
    if(startSample >=0 && startSample < ps_scopeConfig->d_recordLength)
        return {startSample,samples};
    else
        return {0,0};
}

bool FtmwConfig::initialize()
{
    d_currentShift = 0;
    d_lastFom = 0.0;
    d_lastRMS = 0.0;
    double df = d_rfConfig.clockFrequency(RfConfig::DownLO);
    auto sb = d_rfConfig.d_downMixSideband;

    Fid f;
    f.setSpacing(ps_scopeConfig->xIncr());
    f.setSideband(sb);
    f.setProbeFreq(df);
    //divide Vmult by 2^bitShift in case extra padding bits are added
    f.setVMult(ps_scopeConfig->yMult(ps_scopeConfig->d_fidChannel)/pow(2,bitShift()));

    d_fidTemplate = f;
    d_processingPaused = true;

    if(!d_rfConfig.prepareForAcquisition())
    {
        d_errorString = QString("Invalid RF/Chirp configuration.");
        return false;
    }

    p_fidStorage = createStorage(d_number);
    p_fidStorage->start();
    d_lastAutosaveTime = QDateTime::currentDateTime();

#ifdef BC_CUDA
    ps_gpu = std::make_shared<GpuAverager>();
    if(!ps_gpu->initialize(&d_scopeConfig))
    {
        d_errorString = ps_gpu->getErrorString();
        return false;
    }
#endif

    return _init();


}

bool FtmwConfig::advance()
{
    auto s = p_fidStorage->currentSegmentShots();
    auto now = QDateTime::currentDateTime();
    if(d_rfConfig.numSegments() > 1 && d_rfConfig.canAdvance(s))
    {
        d_processingPaused = true;
        d_rfConfig.advanceClockStep();
        p_fidStorage->advance();
        d_lastAutosaveTime = now;
#ifdef BC_CUDA
        ps_gpu->setCurrentData(p_fidStorage->getCurrentFidList());
#endif
        return !isComplete();
    }
    else
    {
        if(d_lastAutosaveTime.addSecs(60) <= now)
        {
            p_fidStorage->save();
            d_lastAutosaveTime = now;
        }
    }

    return false;

}

bool FtmwConfig::setFidsData(const QVector<QVector<qint64> > newList)
{
    FidList l;
    l.reserve(newList.size());
    auto s = p_fidStorage->currentSegmentShots();
    for(int i=0; i<newList.size(); i++)
    {
        Fid f = d_fidTemplate;
        f.setData(newList.at(i));
        auto shots = s+shotIncrement();
        auto p = dynamic_cast<FidPeakUpStorage*>(p_fidStorage.get());
        if(p)
            shots = qMin(p->targetShots(),shots);
        f.setShots(shots);
        l.append(f);
    }

    return p_fidStorage->setFidsData(l);
}

bool FtmwConfig::addFids(const QByteArray rawData)
{
    d_errorString.clear();
    FidList newList;
    if(d_chirpScoringEnabled || d_phaseCorrectionEnabled)
    {
        newList = parseWaveform(rawData);
        if(!preprocessChirp(newList))
            return true;
    }
#ifdef BC_CUDA
    if(!ps_gpu)
        return false;
    if(d_type == Peak_Up)
    {
        //detect if the number of averages has been reset
        if(completedShots() == 0)
            ps_gpu->resetAverage();
        auto p = dynamic_cast<FidPeakUpStorage*>(p_fidStorage.get());
        quint64 ts = 0;
        if(p)
            ts = p->targetShots();
        return setFidsData(ps_gpu->parseAndRollAvg(rawData.constData(),completedShots()+shotIncrement(),ts,d_currentShift));
    }
    else
        return setFidsData(ps_gpu->parseAndAdd(rawData.constData(),d_currentShift));
#else
    if(newList.isEmpty())
        newList = parseWaveform(rawData);
    return p_fidStorage->addFids(newList,d_currentShift);
#endif
}

bool FtmwConfig::addPreAccumulatedFids(const QByteArray &data, quint64 shotCount)
{
    d_errorString.clear();

    int np = ps_scopeConfig->d_recordLength;
    int numRec = ps_scopeConfig->d_numRecords;
    int totalSamples = np * numRec;

    if(data.size() != totalSamples * static_cast<int>(sizeof(qint64)))
    {
        d_errorString = QString("Pre-accumulated data size mismatch: expected %1 bytes, got %2")
                            .arg(totalSamples * sizeof(qint64)).arg(data.size());
        return false;
    }

    const qint64 *src = reinterpret_cast<const qint64*>(data.constData());

    FidList newList;
    newList.reserve(numRec);
    for(int j = 0; j < numRec; ++j)
    {
        QVector<qint64> d(np);
        memcpy(d.data(), src + j*np, np*sizeof(qint64));

        Fid f = d_fidTemplate;
        f.setData(d);
        f.setShots(shotCount);
        newList.append(f);
    }

    if(d_chirpScoringEnabled || d_phaseCorrectionEnabled)
    {
        if(!preprocessChirp(newList))
            return true;
    }

#ifdef BC_CUDA
    // Pre-accumulated data bypasses the GPU averager (which expects raw bytes)
    return p_fidStorage->addFids(newList, d_currentShift);
#else
    return p_fidStorage->addFids(newList, d_currentShift);
#endif
}

void FtmwConfig::hwReady()
{
    d_fidTemplate.setProbeFreq(d_rfConfig.clockFrequency(RfConfig::DownLO));
    d_processingPaused = false;
}

std::shared_ptr<FidStorageBase> FtmwConfig::storage() const
{
    return p_fidStorage;
}


bool FtmwConfig::abort()
{
    return false;
}

void FtmwConfig::cleanupAndSave()
{
    p_fidStorage->finish();
    p_fidStorage->save();
#ifdef BC_CUDA
    ps_gpu.reset();
#endif
}

void FtmwConfig::setWaveformBuffer(WaveformBuffer *buf)
{
    p_waveformBuffer = buf;
}

WaveformBuffer *FtmwConfig::waveformBuffer() const
{
    return p_waveformBuffer;
}

void FtmwConfig::loadFids()
{
    p_fidStorage = createStorage(d_number,d_path);
}

bool FtmwConfig::preprocessChirp(const FidList l)
{
    if(l.isEmpty())
    {
        d_errorString = "Could not parse scope response for preprocessing chirp.";
        return false;
    }

    auto fl = p_fidStorage->getCurrentFidList();
    if(fl.isEmpty())
        return true;

    auto shots = fl.constFirst().shots();
    if(shots < 20ul)
        return true;

    auto r = chirpRange();
    if(r.first < 0 || r.second <= 0)
        return true;

    auto newChirp = l.constFirst().rawData().mid(r.first,r.second);
    auto avgChirp = fl.constFirst().rawData().mid(r.first,r.second);

    if(newChirp.isEmpty() || avgChirp.isEmpty())
        return true;

    bool success = true;
    if(d_chirpScoringEnabled)
    {
        //Calculate chirp RMS
        double newChirpRMS = calculateChirpRMS(newChirp,1ul);

        //Get current RMS
        d_lastRMS = calculateChirpRMS(avgChirp,shots);

        //The chirp is good if its RMS is greater than threshold*currentRMS.
        success = newChirpRMS > d_lastRMS*d_chirpRMSThreshold;
    }

    if(d_phaseCorrectionEnabled)
    {
        int max = 5;
        float thresh = 1.15; // fractional improvement needed to adjust shift
        int shift = d_currentShift;
        auto avgFid = fl.constFirst();
        float fomCenter = calculateFom(newChirp,avgFid,r,shift);
        float fomDown = calculateFom(newChirp,avgFid,r,shift-1);
        float fomUp = calculateFom(newChirp,avgFid,r,shift+1);
        bool done = false;
        while(!done && qAbs(shift-d_currentShift) < max)
        {
            if(fomCenter > fomDown && fomCenter > fomUp)
                done = true;
            else if((fomDown-fomCenter) > (fomUp-fomCenter))
            {
                if(fomDown > thresh*fomCenter)
                {
                    shift--;
                    fomUp = fomCenter;
                    fomCenter = fomDown;
                    fomDown = calculateFom(newChirp,avgFid,r,shift-1);
                }
                else
                    done = true;
            }
            else
            {
                if(fomUp > thresh*fomCenter)
                {
                    shift++;
                    fomDown = fomCenter;
                    fomCenter = fomUp;
                    fomUp = calculateFom(newChirp,avgFid,r,shift+1);
                }
                else
                    done = true;
            }
        }

        if(!done)
        {
            d_errorString = QString("Calculated shift for this FID exceeded maximum permissible shift of %1 points. Fid rejected.").arg(max);
            return false;
        }

        if(qAbs(d_currentShift - shift) > 0)
        {
            if(fomCenter < 0.9*d_lastFom)
            {
                d_errorString = QString("Shot rejected. FOM (%1) is less than 90% of last FOM (%2)").arg(fomCenter,0,'e',2).arg(d_lastFom,0,'e',2);
                return false;
            }

//            d_errorString = QString("Shift changed from %1 to %2. FOMs: (%3, %4, %5)").arg(d_currentShift).arg(shift)
//                            .arg(fomDown,0,'e',2).arg(fomCenter,0,'e',2).arg(fomUp,0,'e',2);
            d_currentShift = shift;
            //        return false;
        }
        if(qAbs(shift) > 50)
        {
            d_errorString = QString("Total shift exceeds maximum range (%1).").arg(50);
            return false;
        }

        d_lastFom = fomCenter;
    }

    return success;

}

float FtmwConfig::calculateFom(const QVector<qint64> vec, const Fid fid, QPair<int, int> range, int trialShift)
{
    //Kahan summation (32 bit precision is sufficient)
    float sum = 0.0;
    float c = 0.0;
    for(int i=0; i<vec.size(); i++)
    {
        if(i+range.first+trialShift >= 0 && i+range.first+trialShift < fid.size())
        {
            float dat = static_cast<float>(fid.atRaw(i+range.first+trialShift))*(static_cast<float>(vec.at(i)));
            float y = dat - c;
            float t = sum + y;
            c = (t-sum) - y;
            sum = t;
        }
    }

    return sum/static_cast<float>(fid.shots());
}

double FtmwConfig::calculateChirpRMS(const QVector<qint64> chirp, quint64 shots)
{
    //Kahan summation
    double sum = 0.0;
    double c = 0.0;
    for(int i=0; i<chirp.size(); i++)
    {
        double dat = static_cast<double>(chirp.at(i)*chirp.at(i))/static_cast<double>(shots*shots);
        double y = dat - c;
        double t = sum + y;
        c = (t-sum) - y;
        sum = t;
    }

    return sqrt(sum);
}

void FtmwConfig::storeValues()
{
    using namespace BC::Store::FTMW;
    store(phase,d_phaseCorrectionEnabled);
    store(chirp,d_chirpScoringEnabled);
    if(d_chirpScoringEnabled)
        store(chirpThresh,d_chirpRMSThreshold);
    if(d_chirpScoringEnabled || d_phaseCorrectionEnabled)
        store(chirpOffset,d_chirpOffsetUs,BC::Unit::us);

    store(ftType,d_type);
    store(objective,d_objective);

    _prepareToSave();
}

void FtmwConfig::retrieveValues()
{
    using namespace BC::Store::FTMW;
    d_phaseCorrectionEnabled = retrieve(phase,false);
    d_chirpScoringEnabled = retrieve(chirp,false);
    d_chirpRMSThreshold = retrieve(chirpThresh,0.0);
    d_chirpOffsetUs = retrieve(chirpOffset,0.0);
    d_objective = retrieve<quint64>(objective);

    //don't need to use retrieved type
    retrieve<FtmwType>(ftType);
    _loadComplete();
}

void FtmwConfig::prepareChildren()
{
    addChild(&d_rfConfig);
    addChild(ps_scopeConfig.get());
}

QString FtmwConfig::objectiveKey() const
{
    return BC::Config::Exp::ftmwType;
}

QVariant FtmwConfig::objectiveData() const
{
    return QVariant::fromValue(d_type);
}
