#include <data/experiment/ftmwconfig.h>

#include <math.h>

#include <QFile>
#include <QtEndian>

#include <data/storage/blackchirpcsv.h>
#include <data/storage/fidpeakupstorage.h>

FtmwConfig::FtmwConfig() : HeaderStorage(BC::Store::FTMW::key)
{
    addChild(&d_rfConfig);
    addChild(&d_scopeConfig);
}

FtmwConfig::~FtmwConfig()
{

}

quint64 FtmwConfig::completedShots() const
{
    return p_fidStorage->completedShots();
}

bool FtmwConfig::processingPaused() const
{
    return d_processingPaused;
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

QVector<qint64> FtmwConfig::extractChirp() const
{
    QVector<qint64> out;
//    FidList dat = fidList();
//    if(!dat.isEmpty())
//    {
//        auto r = chirpRange();
//        if(r.first >= 0 && r.second >= 0)
//            out = dat.constFirst().rawData().mid(r.first, r.second - r.first);
//    }

    return out;
}

QVector<qint64> FtmwConfig::extractChirp(const QByteArray b) const
{
    QVector<qint64> out;
    auto r = chirpRange();
    if(r.first >= 0 && r.second >= 0)
    {
        FidList l = parseWaveform(b);
        if(!l.isEmpty())
            out = l.constFirst().rawData().mid(r.first, r.second - r.first);
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
//    //want to return [first,last) samples for chirp.
//    //TODO: handle multiple chirps
//    auto cc = d_rfConfig.d_chirpConfig;
//    if(cc.chirpList().isEmpty())
//        return qMakePair(-1,-1);

//    if(d_fidList.isEmpty())
//        return qMakePair(-1,-1);

//    //we assume that the scope is triggered at the beginning of the protection pulse

//    double chirpStart = (cc.preChirpGateDelay() + cc.preChirpProtectionDelay() - d_scopeConfig.d_triggerDelayUSec)*1e-6;
//    int startSample = qBound(BC_FTMW_MAXSHIFT,qRound(chirpStart*d_scopeConfig.d_sampleRate) + BC_FTMW_MAXSHIFT,d_fidList.constFirst().size() - BC_FTMW_MAXSHIFT);
//    double chirpEnd = chirpStart + cc.chirpDuration(0)*1e-6;
//    int endSample = qBound(BC_FTMW_MAXSHIFT,qRound(chirpEnd*d_scopeConfig.d_sampleRate) - BC_FTMW_MAXSHIFT,d_fidList.constFirst().size() - BC_FTMW_MAXSHIFT);

//    if(startSample > endSample)
//        qSwap(startSample,endSample);

//    return qMakePair(startSample,endSample);
#pragma message("Implement chirpRange")
    return {0,0};
}

bool FtmwConfig::initialize()
{
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
    d_lastAutosaveTime = QDateTime::currentDateTime();
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

#ifdef BC_CUDA
bool FtmwConfig::setFidsData(const QVector<QVector<qint64> > newList)
{
    FidList l;
    l.reserve(newList.size());
    auto fs = p_fidStorage.get();
    auto s = fs->currentSegmentShots();
    for(int i=0; i<newList.size(); i++)
    {
        Fid f = d_fidTemplate;
        f.setData(newList.at(i));
        f.setShots(s+shotIncrement());
        l.append(f);
    }

    return p_fidStorage.get()->setFidsData(l);
}
#endif

bool FtmwConfig::addFids(const QByteArray rawData, int shift)
{
    FidList newList = parseWaveform(rawData);
    return p_fidStorage.get()->addFids(newList,shift);
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

void FtmwConfig::loadFids(int num, QString path)
{
    p_fidStorage = createStorage(num,path);
}

void FtmwConfig::storeValues()
{
    using namespace BC::Store::FTMW;
    store(phase,d_phaseCorrectionEnabled);
    store(chirp,d_chirpScoringEnabled);
    if(d_chirpScoringEnabled)
        store(chirpThresh,d_chirpRMSThreshold);
    if(d_chirpScoringEnabled || d_phaseCorrectionEnabled)
        store(chirpOffset,d_chirpOffsetUs,QString::fromUtf8("μs"));

    store(type,d_type);

    _prepareToSave();
}

void FtmwConfig::retrieveValues()
{
    using namespace BC::Store::FTMW;
    d_phaseCorrectionEnabled = retrieve(phase,false);
    d_chirpScoringEnabled = retrieve(chirp,false);
    d_chirpRMSThreshold = retrieve(chirpThresh,0.0);
    d_chirpOffsetUs = retrieve(chirpOffset,0.0);

    //don't need to use retrieved type
    retrieve<FtmwType>(type);

    _loadComplete();
}
