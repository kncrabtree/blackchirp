#include <data/experiment/ftmwconfig.h>

#include <math.h>

#include <QFile>
#include <QtEndian>

#include <data/storage/fidsinglestorage.h>

FtmwConfig::~FtmwConfig()
{

}

bool FtmwConfig::isEnabled() const
{
    return d_isEnabled;
}

bool FtmwConfig::isPhaseCorrectionEnabled() const
{
    return d_phaseCorrectionEnabled;
}

bool FtmwConfig::isChirpScoringEnabled() const
{
    return d_chirpScoringEnabled;
}

double FtmwConfig::chirpRMSThreshold() const
{
    return d_chirpRMSThreshold;
}

double FtmwConfig::chirpOffsetUs() const
{
    return d_chirpOffsetUs;
}

BlackChirp::FtmwType FtmwConfig::type() const
{
    return d_type;
}

quint64 FtmwConfig::targetShots() const
{
    return d_targetShots;
}

quint64 FtmwConfig::completedShots() const
{
    return p_fidStorage->completedShots();
}

QDateTime FtmwConfig::targetTime() const
{
    return d_targetTime;
}

//QVector<qint64> FtmwConfig::rawFidList() const
//{
//    int outSize = d_fidList.size();
//    if(outSize == 0)
//        return QVector<qint64>();

//    outSize*=d_fidList.constFirst().size();

//    QVector<qint64> out(outSize);
//    for(int i=0; i<d_fidList.size(); i++)
//    {
//        int offset = i*d_fidList.constFirst().size();
//        for(int j=0; j<d_fidList.at(i).size(); j++)
//            out[offset+j] = d_fidList.at(i).atRaw(j);
//    }

//    return out;
//}

//QList<FidList> FtmwConfig::multiFidList() const
//{
//    return d_multiFidStorage;
//}

const FtmwDigitizerConfig& FtmwConfig::scopeConfig() const
{
    return d_scopeConfig;
}

RfConfig FtmwConfig::rfConfig() const
{
    return d_rfConfig;
}

ChirpConfig FtmwConfig::chirpConfig(int num) const
{
    return d_rfConfig.getChirpConfig(num);
}

Fid FtmwConfig::fidTemplate() const
{
    return d_fidTemplate;
}

bool FtmwConfig::processingPaused() const
{
    return d_processingPaused;
}

int FtmwConfig::numFrames() const
{
    return d_scopeConfig.d_numRecords;
}

int FtmwConfig::numSegments() const
{
    return d_rfConfig.numSegments();
}

quint64 FtmwConfig::shotIncrement() const
{
    quint64 increment = 1;
    if(d_scopeConfig.d_blockAverage)
        increment *= d_scopeConfig.d_numAverages;
    if(d_scopeConfig.d_multiRecord)
        increment *= d_scopeConfig.d_numRecords;

    return increment;
}

FidList FtmwConfig::parseWaveform(const QByteArray b) const
{

    int np = d_scopeConfig.d_recordLength;
    FidList out;
    //read raw data into vector in 64 bit integer form
    for(int j=0;j<numFrames();j++)
    {
        QVector<qint64> d(np);

        for(int i=0; i<np;i++)
        {
            qint64 dat = 0;

            /*
            for(int k = 0; k<scopeConfig().bytesPerPoint; k++)
            {
                int thisIndex = k;
                if(scopeConfig().byteOrder == QDataStream::BigEndian)
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

                if(d_scopeConfig.d_byteOrder == QDataStream::BigEndian)
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

                if(d_scopeConfig.d_byteOrder == QDataStream::BigEndian)
                    y = qFromBigEndian(y);
                else
                    y = qFromLittleEndian(y);

                dat = (static_cast<qint64>(y));
            }


            //in peak up mode, add 8 bits of padding so that there are empty bits to fill when
            //the rolling average kicks in.
            //Note that this could lead to overflow problems if bytesPerPoint = 4 and the # of averages is large
            if(type() == BlackChirp::FtmwPeakUp)
                dat = dat << 8;

            d[i] = dat;
        }

        Fid f = fidTemplate();
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

QString FtmwConfig::errorString() const
{
    return d_errorString;
}

double FtmwConfig::ftMinMHz() const
{
    double sign = 1.0;
    if(rfConfig().downMixSideband() == BlackChirp::LowerSideband)
        sign = -1.0;
    double lo = rfConfig().clockFrequency(BlackChirp::DownConversionLO);
    double lastFreq = lo + sign*ftNyquistMHz();
    return qMin(lo,lastFreq);
}

double FtmwConfig::ftMaxMHz() const
{
    double sign = 1.0;
    if(rfConfig().downMixSideband() == BlackChirp::LowerSideband)
        sign = -1.0;
    double lo = rfConfig().clockFrequency(BlackChirp::DownConversionLO);
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
//    auto cc = rfConfig().getChirpConfig();
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

bool FtmwConfig::writeFids(int num, QString path, int snapNum) const
{
    (void)num;
    (void)path;
    (void)snapNum;
#pragma message("Figure out writeFids")
    return true;
//    if(!d_multipleFidLists)
//    {
//        QFile fid(BlackChirp::getExptFile(num,BlackChirp::FidFile,path,snapNum));
//        if(fid.open(QIODevice::WriteOnly))
//        {
//            QDataStream d(&fid);
//            d << Fid::magicString();
//            d << d_fidList;
//            fid.close();
//            return true;
//        }
//        else
//            return false;
//    }
//    else
//    {
//        QFile fid(BlackChirp::getExptFile(num,BlackChirp::MultiFidFile,path,snapNum));
//        if(fid.open(QIODevice::WriteOnly))
//        {
//            QDataStream d(&fid);
//            d << Fid::magicString();
//            d << d_multiFidStorage;
//            fid.close();
//            return true;
//        }
//        else
//            return false;
//    }
}

bool FtmwConfig::initialize()
{
    double df = rfConfig().clockFrequency(BlackChirp::DownConversionLO);
    auto sb = rfConfig().downMixSideband();

    Fid f(d_scopeConfig.xIncr(),df,QVector<qint64>(0),sb,d_scopeConfig.yMult(d_scopeConfig.d_fidChannel),1);

    //in peak up mode, data points will be shifted by 8 bits (x256), so the multiplier
    //needs to decrease by a factor of 256
    if(type() == BlackChirp::FtmwPeakUp)
        f.setVMult(f.vMult()/256.0);
    d_fidTemplate = f;

    if(!d_rfConfig.prepareForAcquisition(type()))
    {
        d_errorString = QString("Invalid RF/Chirp configuration.");
        return false;
    }

    if(type() == BlackChirp::FtmwLoScan || type() == BlackChirp::FtmwDrScan)
        d_targetShots = d_rfConfig.totalShots();

    p_fidStorage = std::make_shared<FidSingleStorage>(QString(""),scopeConfig().d_numRecords);

//    d_completedShots = 0;

    return true;


}

void FtmwConfig::setEnabled(bool en)
{
    d_isEnabled = en;
}

void FtmwConfig::setPhaseCorrectionEnabled(bool enabled)
{
    d_phaseCorrectionEnabled = enabled;
}

void FtmwConfig::setChirpScoringEnabled(bool enabled)
{
    d_chirpScoringEnabled = enabled;
}

void FtmwConfig::setChirpRMSThreshold(double t)
{
    d_chirpRMSThreshold = t;
}

void FtmwConfig::setChirpOffsetUs(double o)
{
    d_chirpOffsetUs = o;
}

void FtmwConfig::setFidTemplate(const Fid f)
{
    d_fidTemplate = f;
}

void FtmwConfig::setType(const BlackChirp::FtmwType type)
{
    d_type = type;
}

void FtmwConfig::setTargetShots(const qint64 target)
{
    d_targetShots = target;
}

bool FtmwConfig::advance()
{
    auto fs = p_fidStorage.get();
//    d_completedShots = fs->completedShots();
    auto s = fs->currentSegmentShots();
    if(d_rfConfig.canAdvance(s))
    {
        d_processingPaused = true;
        //adjust number of completed shots in case the increment does not go evenly into
        //segment shots
//        d_completedShots = d_rfConfig.completedSegmentShots();
        return !isComplete();

    }

    return false;

}

void FtmwConfig::setTargetTime(const QDateTime time)
{
    d_targetTime = time;
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
        Fid f = fidTemplate();
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

//void FtmwConfig::addFids(const FtmwConfig other)
//{
//    if(d_multipleFidLists)
//    {
//        auto l = other.multiFidList();
//        for(int i=0; i<l.size(); i++)
//        {
//            if(d_multiFidStorage.size() == i)
//                d_multiFidStorage.append(l.at(i));
//            else
//            {
//                if(d_multiFidStorage.at(i).size() != l.at(i).size())
//                    d_multiFidStorage[i] = l.at(i);
//                else
//                {
//                    for(int j=0; j<d_multiFidStorage.at(i).size(); j++)
//                        d_multiFidStorage[i][j] += l.at(i).at(j);
//                }
//            }
//        }
//    }
//    else
//    {
//        auto l = other.fidList();
//        for(int i=0; i<l.size(); i++)
//        {
//            if(d_fidList.size() == i)
//                d_fidList.append(l.at(i));
//            else
//                d_fidList[i] += l.at(i);
//        }
//    }
//}

bool FtmwConfig::subtractFids(const FtmwConfig other)
{
    (void)other;
#pragma message("Figure out what to do with subtractFids")
//    if(!d_multipleFidLists)
//    {
//        auto otherList = other.fidList();

//        if(otherList.size() != d_fidList.size())
//            return false;

//        for(int i=0; i<otherList.size(); i++)
//        {
//            if(otherList.at(i).size() != d_fidList.at(i).size())
//                return false;

//            if(otherList.at(i).shots() >= d_fidList.at(i).shots())
//                return false;
//        }

//        for(int i=0; i<d_fidList.size(); i++)
//            d_fidList[i] -= otherList.at(i);

//    }
//    else
//    {
//        auto otherList = other.multiFidList();

//        for(int i=0; i<d_multiFidStorage.size(); i++)
//        {
//            if(i >= otherList.size())
//                continue;

//            if(otherList.at(i).size() != d_multiFidStorage.at(i).size() || d_multiFidStorage.at(i).isEmpty())
//                return false;

//            //if numbers of shots are equal, then no new data have been added for this chunk.
//            //Write an empty list of fids.
//            //Otherwise, get the difference.
//            if(otherList.at(i).constFirst().shots() == d_multiFidStorage.at(i).constFirst().shots())
//                d_multiFidStorage[i] = FidList();
//            else if(otherList.at(i).constFirst().shots() < d_multiFidStorage.at(i).constFirst().shots())
//            {
//                for(int j=0; j<d_multiFidStorage.at(i).size(); j++)
//                    d_multiFidStorage[i][j] -= otherList.at(i).at(j);
//            }
//            else
//                return false;
//        }
//    }

    return true;
}

void FtmwConfig::resetFids()
{
    p_fidStorage.get()->reset();
//    d_fidList.clear();
//    d_completedShots = 0;
}

void FtmwConfig::setScopeConfig(const FtmwDigitizerConfig &other)
{
    d_scopeConfig = other;
}

void FtmwConfig::setRfConfig(const RfConfig other)
{
    d_rfConfig = other;
}

void FtmwConfig::hwReady()
{
    d_fidTemplate.setProbeFreq(rfConfig().clockFrequency(BlackChirp::DownConversionLO));
    d_processingPaused = false;
}

int FtmwConfig::perMilComplete() const
{
    if(indefinite())
        return 0;

    return static_cast<int>(floor(static_cast<double>(completedShots())/static_cast<double>(targetShots()) * 1000.0));
}

bool FtmwConfig::indefinite() const
{
    if(type() == BlackChirp::FtmwForever)
        return true;

    return false;
}

void FtmwConfig::finalizeSnapshots(int num, QString path)
{
    //write current fid or mfd file
    //load snap file; get number of snapshots
    //delete all snapshot files
    //delete snap file
    //recalculate completed shots
    writeFids(num,path);

    QFile snp(BlackChirp::getExptFile(num,BlackChirp::SnapFile,path));
    int snaps = 0;
    if(snp.open(QIODevice::ReadOnly))
    {
        QByteArrayList l;
        while(!snp.atEnd())
        {
            QByteArray line = snp.readLine();
            if(!line.isEmpty() && !line.startsWith("fid") && !line.startsWith("mfd"))
                l.append(line);
            else
            {
                auto ll = QString(line).split(QString("\t"));
                if(ll.size() >= 2)
                    snaps = ll.at(1).trimmed().toInt();
            }
        }
        snp.close();

        //if there's anything left (eg LIF snapshots), rewrite the file with those
        if(!l.isEmpty())
        {
            snp.open(QIODevice::WriteOnly);
            while(!l.isEmpty())
                snp.write(l.takeFirst());
            snp.close();
        }
        else
            snp.remove();
    }

    (void)snaps;
#pragma message("finalizeSnapshots behavior - what to do?")

//    for(int i=0; i<snaps; i++)
//    {
//        if(!d_multipleFidLists)
//        {
//            QFile snap(BlackChirp::getExptFile(num,BlackChirp::FidFile,path,i));
//            if(snap.exists())
//                snap.remove();
//        }
//        else
//        {
//            QFile snap(BlackChirp::getExptFile(num,BlackChirp::MultiFidFile,path,i));
//            if(snap.exists())
//                snap.remove();
//        }
//    }

//    qint64 ts = 0;
//    if(d_multipleFidLists)
//    {
//        for(int i=0; i<d_multiFidStorage.size(); i++)
//            ts += d_multiFidStorage.at(i).constFirst().shots();
//    }
//    else
//    {
//        for(int i=0; i<d_fidList.size(); i++)
//            ts += d_fidList.constFirst().shots();
//    }

//    d_completedShots = ts;


}

std::shared_ptr<FidStorageBase> FtmwConfig::storage()
{
    return p_fidStorage;
}

bool FtmwConfig::isComplete() const
{
    if(!isEnabled())
        return true;

    switch(type())
    {
    case BlackChirp::FtmwTargetShots:
    case BlackChirp::FtmwLoScan:
    case BlackChirp::FtmwDrScan:
        return completedShots() >= targetShots();
        break;
    case BlackChirp::FtmwTargetTime:
        return QDateTime::currentDateTime() >= targetTime();
        break;
    case BlackChirp::FtmwForever:
    case BlackChirp::FtmwPeakUp:
    default:
        return false;
        break;
    }

    //not reached
    return false;
}

bool FtmwConfig::abort()
{
    return false;
}

QMap<QString, QPair<QVariant, QString> > FtmwConfig::headerMap() const
{
    QMap<QString, QPair<QVariant, QString> > out;

    QString prefix = QString("FtmwConfig");
    QString empty = QString("");

    out.insert(prefix+QString("Enabled"),qMakePair(isEnabled(),empty));
    if(!isEnabled())
        return out;

    out.insert(prefix+QString("Type"),qMakePair((int)type(),empty));
    if(type() == BlackChirp::FtmwTargetShots)
        out.insert(prefix+QString("TargetShots"),qMakePair(targetShots(),empty));
    if(type() == BlackChirp::FtmwTargetTime)
        out.insert(prefix+QString("TargetTime"),qMakePair(targetTime(),empty));
//    out.insert(prefix+QString("CompletedShots"),qMakePair(d_completedShots,empty));
    out.insert(prefix+QString("FidVMult"),qMakePair(QString::number(fidTemplate().vMult(),'g',12),QString("V")));
    out.insert(prefix+QString("PhaseCorrection"),qMakePair(d_phaseCorrectionEnabled,QString("")));
    out.insert(prefix+QString("ChirpScoring"),qMakePair(d_chirpScoringEnabled,QString("")));
    out.insert(prefix+QString("ChirpRMSThreshold"),qMakePair(QString::number(d_chirpRMSThreshold,'f',3),empty));
    out.insert(prefix+QString("ChirpOffset"),qMakePair(QString::number(d_chirpOffsetUs,'f',4),QString::fromUtf16(u" μs")));


//    out.unite(scopeConfig.headerMap());
    out.unite(d_rfConfig.headerMap());

    return out;

}

void FtmwConfig::loadFids(const int num, const QString path)
{
    (void)num;
    (void)path;
#pragma message("How to deal with loading FIDs?")
//    QFile fid(BlackChirp::getExptFile(num,BlackChirp::FidFile,path));
//    if(fid.open(QIODevice::ReadOnly))
//    {
//        QDataStream d(&fid);
//        QByteArray magic;
//        d >> magic;
//        if(magic.startsWith("BCFID"))
//        {
//            if(magic.endsWith("v1.0"))
//            {
//                FidList dat;
//                d >> dat;
//                d_fidList = dat;
//                if(!dat.isEmpty())
//                    d_fidTemplate = dat.constFirst();
//                d_fidTemplate.setData(QVector<qint64>());
//            }
//        }
//        fid.close();
//    }
    
//    QFile mfd(BlackChirp::getExptFile(num,BlackChirp::MultiFidFile,path));
//    if(mfd.open(QIODevice::ReadOnly))
//    {
//        d_multipleFidLists = true;
//        QDataStream d(&mfd);
//        QByteArray magic;
//        d >> magic;
//        if(magic.startsWith("BCFID"))
//        {
//            if(magic.endsWith("v1.0"))
//            {
//                QList<FidList> dat;
//                d >> dat;
//                d_multiFidStorage = dat;
//                if(!dat.isEmpty() && !dat.constFirst().isEmpty())
//                    d_fidTemplate = dat.constFirst().constFirst();
//                d_fidTemplate.setData(QVector<qint64>());
//            }
//        }
//        mfd.close();
//    }

//    if(d_fidList.isEmpty() && d_multiFidStorage.isEmpty())
//    {
//        //try to reconstruct from snapshots, if any
//        QFile snp(BlackChirp::getExptFile(num,BlackChirp::SnapFile,path));
//        if(snp.exists() && snp.open(QIODevice::ReadOnly))
//        {
//            bool parseSuccess = false;
//            bool multiFid = false;
//            int numSnaps = 0;
//            while(!snp.atEnd())
//            {
//                QString line = snp.readLine();
//                if(line.startsWith(QString("fid")) || line.startsWith(QString("mfd")))
//                {
//                    if(line.startsWith(QString("mfd")))
//                        multiFid = true;

//                    QStringList l = line.split(QString("\t"));
//                    bool ok = false;
//                    int n = l.constLast().trimmed().toInt(&ok);
//                    if(ok)
//                    {
//                        parseSuccess = true;
//                        numSnaps = n;
//                        break;
//                    }
                    
//                }
                
//            }
            
//            QList<int> snaps;
//            for(int i=0; i<numSnaps; i++)
//                snaps << i;

//            if(parseSuccess && numSnaps > 0)
//            {
//                d_multipleFidLists = multiFid;
//                loadFidsFromSnapshots(num,path,snaps);
//            }

//            snp.close();
//        }
//    }

//    qint64 ts = 0;
//    if(d_multipleFidLists)
//    {
//        for(int i=0; i<d_multiFidStorage.size(); i++)
//            ts += d_multiFidStorage.at(i).constFirst().shots();
//    }
//    else
//    {
//        for(int i=0; i<d_fidList.size(); i++)
//            ts += d_fidList.constFirst().shots();
//    }

////    if(scopeConfig.fastFrameEnabled && scopeConfig.summaryFrame && !scopeConfig().manualFrameAverage)
////        ts *= scopeConfig.numFrames;

//    d_completedShots = ts;

    
}

void FtmwConfig::loadFidsFromSnapshots(const int num, const QString path, const QList<int> snaps)
{
    (void)num;
    (void)path;
    (void)snaps;
#pragma message("Snapshot issue")
//    if(d_multipleFidLists)
//    {
//        d_multiFidStorage.clear();

//        for(int i=0; i<snaps.size(); i++)
//        {
//            QFile mfd(BlackChirp::getExptFile(num,BlackChirp::MultiFidFile,path,snaps.at(i)));
//            if(mfd.open(QIODevice::ReadOnly))
//            {
//                QDataStream d(&mfd);
//                QByteArray magic;
//                d >> magic;
//                if(magic.startsWith("BCFID"))
//                {
//                    if(magic.endsWith("v1.0"))
//                    {
//                        QList<FidList> dat;
//                        d >> dat;
//                        if(d_multiFidStorage.isEmpty())
//                        {
//                            d_multiFidStorage = dat;
//                            if(!dat.isEmpty() && !dat.constFirst().isEmpty())
//                                d_fidTemplate = dat.constFirst().constFirst();
//                            d_fidTemplate.setData(QVector<qint64>());
//                        }
//                        else
//                        {
//                            for(int j=0; j<dat.size(); j++)
//                            {
//                                if(j == d_multiFidStorage.size())
//                                    d_multiFidStorage << dat.at(j);
//                                else
//                                {
//                                    for(int k=0; k<dat.at(j).size() && k<d_multiFidStorage.at(j).size(); k++)
//                                        d_multiFidStorage[j][k] += dat.at(j).at(k);
//                                }
//                            }
//                        }
//                    }
//                }
//                mfd.close();

//            }
//        }
//    }
//    else
//    {
//        d_fidList.clear();
//        for(int i=0; i<snaps.size(); i++)
//        {
//            QFile fid(BlackChirp::getExptFile(num,BlackChirp::FidFile,path,snaps.at(i)));
//            if(fid.open(QIODevice::ReadOnly))
//            {
//                QDataStream d(&fid);
//                QByteArray magic;
//                d >> magic;
//                if(magic.startsWith("BCFID"))
//                {
//                    if(magic.endsWith("v1.0"))
//                    {
//                        FidList dat;
//                        d >> dat;
//                        if(d_fidList.isEmpty())
//                        {
//                            d_fidList= dat;
//                            if(!dat.isEmpty())
//                                d_fidTemplate = dat.constFirst();
//                            d_fidTemplate.setData(QVector<qint64>());
//                        }
//                        else
//                        {
//                            for(int j=0; j<d_fidList.size() && j<dat.size(); j++)
//                                d_fidList[j] += dat.at(j);
//                        }
//                    }
//                }

//                fid.close();
//            }
//        }
//    }
}

void FtmwConfig::parseLine(const QString key, const QVariant val)
{
//    if(key.startsWith(QString("FtmwScope")))
//    {
//        if(key.endsWith(QString("FidChannel")))
//            scopeConfig.fidChannel = val.toInt();
//        if(key.endsWith(QString("VerticalScale")))
//            scopeConfig.vScale = val.toDouble();
//        if(key.endsWith(QString("VerticalOffset")))
//            scopeConfig.vOffset = val.toDouble();
//        if(key.endsWith(QString("TriggerChannel")))
//        {
//            if(val.toString().contains(QString("AuxIn")))
//                scopeConfig.trigChannel = 0;
//            else
//                scopeConfig.trigChannel = val.toInt();
//        }
//        if(key.endsWith(QString("TriggerDelay")))
//            scopeConfig.trigDelay = val.toDouble();
//        if(key.endsWith(QString("TriggerLevel")))
//            scopeConfig.trigLevel = val.toDouble();
//        if(key.endsWith(QString("TriggerSlope")))
//        {
//            if(val.toString().contains(QString("Rising")))
//                scopeConfig.slope = BlackChirp::RisingEdge;
//            else
//                scopeConfig.slope = BlackChirp::FallingEdge;
//        }
//        if(key.endsWith(QString("SampleRate")))
//            scopeConfig.sampleRate = val.toDouble()*1e9;
//        if(key.endsWith(QString("RecordLength")))
//            scopeConfig.recordLength = val.toInt();
//        if(key.endsWith(QString("FastFrame")))
//            scopeConfig.fastFrameEnabled = val.toBool();
//        if(key.endsWith(QString("SummaryFrame")))
//            scopeConfig.summaryFrame = val.toBool();
//        if(key.endsWith(QString("BlockAverage")))
//            scopeConfig.blockAverageEnabled = val.toBool();
//        if(key.endsWith(QString("NumAverages")))
//            scopeConfig.numAverages = val.toInt();
//        if(key.endsWith(QString("BytesPerPoint")))
//            scopeConfig.bytesPerPoint = val.toInt();
//        if(key.endsWith(QString("NumFrames")))
//            scopeConfig.numFrames = val.toInt();
//        if(key.endsWith(QString("ByteOrder")))
//        {
//            if(val.toString().contains(QString("BigEndian")))
//                scopeConfig.byteOrder = QDataStream::BigEndian;
//            else
//                scopeConfig.byteOrder = QDataStream::LittleEndian;
//        }
//    }
    if(key.startsWith(QString("FtmwConfig")))
    {
        if(key.endsWith(QString("Enabled")))
            d_isEnabled = val.toBool();
        if(key.endsWith(QString("Type")))
            d_type = (BlackChirp::FtmwType)val.toInt();
        if(key.endsWith(QString("TargetShots")))
            d_targetShots = val.toInt();
//        if(key.endsWith(QString("CompletedShots")))
//            d_completedShots = val.toInt();
        if(key.endsWith(QString("TargetTime")))
            d_targetTime = val.toDateTime();
        if(key.endsWith(QString("PhaseCorrection")))
            d_phaseCorrectionEnabled = val.toBool();
        if(key.endsWith(QString("ChirpScoring")))
            d_chirpScoringEnabled = val.toBool();
        if(key.endsWith(QString("ChirpRMSThreshold")))
            d_chirpRMSThreshold = val.toDouble();
        if(key.endsWith(QString("ChirpOffset")))
            d_chirpOffsetUs = val.toDouble();
    }
    else if(key.startsWith(QString("RfConfig")))
        d_rfConfig.parseLine(key,val);
}

void FtmwConfig::loadChirps(const int num, const QString path)
{
    ///TODO: Figure out the future of loading chirps from disk
    d_rfConfig.addChirpConfig(ChirpConfig(num,path));
}

void FtmwConfig::loadClocks(const int num, const QString path)
{
    d_rfConfig.loadClockSteps(num,path);
}

void FtmwConfig::saveToSettings() const
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("lastFtmwConfig"));

    s.setValue(QString("mode"),static_cast<int>(type()));
    s.setValue(QString("targetShots"),targetShots());
    s.setValue(QString("targetTime"),QDateTime::currentDateTime().msecsTo(targetTime()));
    s.setValue(QString("phaseCorrection"),isPhaseCorrectionEnabled());
    s.setValue(QString("chirpScoring"),isChirpScoringEnabled());
    s.setValue(QString("chirpOffsetUs"),chirpOffsetUs());
    s.setValue(QString("chirpRMSThreshold"),chirpRMSThreshold());

//    s.setValue(QString("fidChannel"),scopeConfig().fidChannel);
//    s.setValue(QString("vScale"),scopeConfig().vScale);
//    s.setValue(QString("triggerChannel"),scopeConfig().trigChannel);
//    s.setValue(QString("triggerDelay"),scopeConfig().trigDelay);
//    s.setValue(QString("triggerLevel"),scopeConfig().trigLevel);
//    s.setValue(QString("triggerSlope"),static_cast<int>(scopeConfig().slope));
//    s.setValue(QString("sampleRate"),scopeConfig().sampleRate);
//    s.setValue(QString("recordLength"),scopeConfig().recordLength);
//    s.setValue(QString("bytesPerPoint"),scopeConfig().bytesPerPoint);
//    s.setValue(QString("fastFrame"),scopeConfig().fastFrameEnabled);
//    s.setValue(QString("numFrames"),scopeConfig().numFrames);
//    s.setValue(QString("summaryFrame"),scopeConfig().summaryFrame);
//    s.setValue(QString("blockAverage"),scopeConfig().blockAverageEnabled);
//    s.setValue(QString("numAverages"),scopeConfig().numAverages);

    s.endGroup();

    d_rfConfig.saveToSettings();


}

FtmwConfig FtmwConfig::loadFromSettings()
{
    FtmwConfig out;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lastFtmwConfig"));

    out.setType(static_cast<BlackChirp::FtmwType>(s.value(QString("mode"),0).toInt()));
    out.setTargetShots(s.value(QString("targetShots"),10000).toInt());
    out.setTargetTime(QDateTime::currentDateTime().addMSecs(s.value(QString("targetTime"),3600000).toInt()));
    out.setPhaseCorrectionEnabled(s.value(QString("phaseCorrection"),false).toBool());
    out.setChirpScoringEnabled(s.value(QString("chirpScoring"),false).toBool());
    out.setChirpRMSThreshold(s.value(QString("chirpRMSThreshold"),0.0).toDouble());
    out.setChirpOffsetUs(s.value(QString("chirpOffsetUs"),-1.0).toDouble());

    BlackChirp::FtmwScopeConfig sc;
    sc.fidChannel = s.value(QString("fidChannel"),1).toInt();
    sc.vScale = s.value(QString("vScale"),0.02).toDouble();
    sc.trigChannel = s.value(QString("triggerChannel"),4).toInt();
    sc.trigDelay = s.value(QString("triggerDelay"),0.0).toDouble();
    sc.trigLevel = s.value(QString("triggerLevel"),0.35).toDouble();
    sc.slope = static_cast<BlackChirp::ScopeTriggerSlope>(s.value(QString("triggerSlope"),0).toInt());
    sc.sampleRate = s.value(QString("sampleRate"),50e9).toDouble();
    sc.recordLength = s.value(QString("recordLength"),750000).toInt();
    sc.bytesPerPoint = s.value(QString("bytesPerPoint"),1).toInt();
    sc.fastFrameEnabled = s.value(QString("fastFrame"),false).toBool();
    sc.numFrames = s.value(QString("numFrames"),1).toInt();
    sc.summaryFrame = s.value(QString("summaryFrame"),false).toBool();
    sc.blockAverageEnabled = s.value(QString("blockAverage"),false).toBool();
    sc.numAverages = s.value(QString("numAverages"),1).toInt();
//    out.setScopeConfig(sc);
    out.setRfConfig(RfConfig::loadFromSettings());

    return out;
}

