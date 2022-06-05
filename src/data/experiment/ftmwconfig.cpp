#include <data/experiment/ftmwconfig.h>

#include <math.h>

#include <QFile>
#include <QtEndian>

#include <data/storage/blackchirpcsv.h>
#include <data/storage/fidpeakupstorage.h>

FtmwConfig::FtmwConfig() : HeaderStorage(BC::Store::FTMW::key)
{
}

FtmwConfig::~FtmwConfig()
{

}

quint64 FtmwConfig::shotIncrement() const
{
    quint64 increment = 1;
    if(d_scopeConfig.d_blockAverage)
        increment *= d_scopeConfig.d_numAverages;

    return increment;
}

FidList FtmwConfig::parseWaveform(const QByteArray b) const
{

    int np = d_scopeConfig.d_recordLength;
    auto shots = shotIncrement();
    FidList out;
    //read raw data into vector in 64 bit integer form
    for(int j=0;j<d_scopeConfig.d_numRecords;j++)
    {
        QVector<qint64> d(np);

        for(int i=0; i<np;i++)
        {
            qint64 dat = 0;

            /*
            for(int k = 0; k<scopeConfig().bytesPerPoint; k++)
            {
                int thisIndex = k;
                if(scopeConfig().byteOrder == DigitizerConfig::BigEndian)
                    thisIndex = scopeConfig().bytesPerPoint - k;

                dat |= (static_cast<quint8>(b.at(scopeConfig().bytesPerPoint*(j*np+i)+thisIndex)) << (8*k));
            }
            //check for the sign bit on the most significant byte, and carry out sign extension if necessary
            if(dat | (128 << (scopeConfig().bytesPerPoint-1)))
                dat &= Q_INT64_C(0xffffffffffffffff);

            dat += static_cast<qint64>(scopeConfig().yOff);
            */


            if(d_scopeConfig.d_bytesPerPoint == 1)
            {
                char y = b.at(j*np+i);
                dat = static_cast<qint64>(y);
            }
            else if(d_scopeConfig.d_bytesPerPoint == 2)
            {
                auto y1 = static_cast<quint8>(b.at(2*(j*np+i)));
                auto y2 = static_cast<quint8>(b.at(2*(j*np+i) + 1));

                qint16 y = 0;
                y |= y1;
                y |= (y2 << 8);

                if(d_scopeConfig.d_byteOrder == DigitizerConfig::BigEndian)
                    y = qFromBigEndian(y);
                else
                    y = qFromLittleEndian(y);

                dat = (static_cast<qint64>(y));
            }
            else
            {
                auto y1 = static_cast<quint8>(b.at(4*(j*np+i)));
                auto y2 = static_cast<quint8>(b.at(4*(j*np+i) + 1));
                auto y3 = static_cast<quint8>(b.at(4*(j*np+i) + 2));
                auto y4 = static_cast<quint8>(b.at(4*(j*np+i) + 3));

                qint32 y = 0;
                y |= y1;
                y |= (y2 << 8);
                y |= (y3 << 16);
                y |= (y4 << 24);

                if(d_scopeConfig.d_byteOrder == DigitizerConfig::BigEndian)
                    y = qFromBigEndian(y);
                else
                    y = qFromLittleEndian(y);

                dat = (static_cast<qint64>(y));
            }

            //"Undo" averaging that was done by the device
            //Ok to do this if statement in the loop; the compiler will optimize it
            if(shots > 1)
                dat *= shots;

            //some modes (eg peakup) may add additional padding bits for averaging
            dat = dat << bitShift();

            d[i] = dat;
        }

        Fid f = d_fidTemplate;
        f.setData(d);
        f.setShots(shotIncrement());

        out.append(f);
    }

    return out;
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
    return d_scopeConfig.d_sampleRate/(1e6*2.0);
}

double FtmwConfig::fidDurationUs() const
{
    double sr = d_scopeConfig.d_sampleRate;
    double rl = static_cast<double>(d_scopeConfig.d_recordLength);

    return rl/sr*1e6;
}

QPair<int, int> FtmwConfig::chirpRange() const
{
    //compute chirp duration in samples (only use first chirp if there are multiple)
    auto dur = d_rfConfig.d_chirpConfig.chirpDurationUs(0);
    if(d_scopeConfig.d_sampleRate <= 0.0)
        return {0,0};

    int samples = dur*1e-6*d_scopeConfig.d_sampleRate;

    //we assume that the scope is triggered at the beginning of the protection pulse
    //unless the user has specified a custom start time
    double startUs = 0.0;
    if(d_chirpOffsetUs < 0.0)
    {
        auto cc = d_rfConfig.d_chirpConfig;
        startUs = cc.preChirpGateDelay() + cc.preChirpProtectionDelay() - d_scopeConfig.d_triggerDelayUSec;
    }
    else
        startUs = d_chirpOffsetUs;

    int startSample = startUs*1e-6*d_scopeConfig.d_sampleRate;
    if(startSample >=0 && startSample < d_scopeConfig.d_recordLength)
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
    f.setSpacing(d_scopeConfig.xIncr());
    f.setSideband(sb);
    f.setProbeFreq(df);
    //divide Vmult by 2^bitShift in case extra padding bits are added
    f.setVMult(d_scopeConfig.yMult(d_scopeConfig.d_fidChannel)/pow(2,bitShift()));

    d_fidTemplate = f;

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

void FtmwConfig::setScopeConfig(const FtmwDigitizerConfig &other)
{
    d_scopeConfig = other;
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

void FtmwConfig::cleanup()
{
    p_fidStorage->finish();
#ifdef BC_CUDA
    ps_gpu.reset();
#endif
}

void FtmwConfig::loadFids(int num, QString path)
{
    p_fidStorage = createStorage(num,path);
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

    store(type,d_type);
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
    retrieve<FtmwType>(type);
    _loadComplete();
}

void FtmwConfig::prepareChildren()
{
    addChild(&d_rfConfig);
    addChild(&d_scopeConfig);
}
