#include <data/experiment/ftmwconfig.h>

#include <math.h>

#include <QFile>
#include <QtEndian>

#include <data/storage/fidsinglestorage.h>
#include <data/storage/fidpeakupstorage.h>

FtmwConfig::FtmwConfig() : HeaderStorage(BC::Store::FTMW::key)
{

}

FtmwConfig::~FtmwConfig()
{

}

quint64 FtmwConfig::completedShots() const
{
    return p_fidStorage->completedShots();
}

QDateTime FtmwConfig::targetTime() const
{
    return d_targetTime;
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
    if(d_scopeConfig.d_multiRecord)
        increment *= d_scopeConfig.d_numRecords;

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
            if(d_type == Peak_Up)
                dat = dat << 8;

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
//    auto cc = d_rfConfig.getChirpConfig();
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
    double df = d_rfConfig.clockFrequency(RfConfig::DownLO);
    auto sb = d_rfConfig.d_downMixSideband;

    Fid f(d_scopeConfig.xIncr(),df,QVector<qint64>(0),sb,d_scopeConfig.yMult(d_scopeConfig.d_fidChannel),1);

    //in peak up mode, data points will be shifted by 8 bits (x256), so the multiplier
    //needs to decrease by a factor of 256
    if(d_type == Peak_Up)
        f.setVMult(f.vMult()/256.0);
    d_fidTemplate = f;

    if(!d_rfConfig.prepareForAcquisition())
    {
        d_errorString = QString("Invalid RF/Chirp configuration.");
        return false;
    }

    if(d_type == LO_Scan || d_type == DR_Scan)
        d_targetShots = d_rfConfig.totalShots();
    if(d_type == Target_Duration)
        d_targetTime = QDateTime::currentDateTime().addSecs(d_duration*60);

    p_fidStorage = std::make_shared<FidSingleStorage>(QString(""),d_scopeConfig.d_numRecords);

    return true;


}

bool FtmwConfig::advance()
{
    auto fs = p_fidStorage.get();
    auto s = fs->currentSegmentShots();
    if(d_rfConfig.canAdvance(s))
    {
        d_processingPaused = true;
        d_rfConfig.advanceClockStep();
        return !isComplete();

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

void FtmwConfig::setScopeConfig(const FtmwDigitizerConfig &other)
{
    d_scopeConfig = other;
}

void FtmwConfig::hwReady()
{
    d_fidTemplate.setProbeFreq(d_rfConfig.clockFrequency(RfConfig::DownLO));
    d_processingPaused = false;
}

int FtmwConfig::perMilComplete() const
{
    if(indefinite())
        return 0;

    return static_cast<int>(floor(static_cast<double>(completedShots())/static_cast<double>(d_targetShots) * 1000.0));
}

bool FtmwConfig::indefinite() const
{
    if(d_type == Forever)
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

std::shared_ptr<FidStorageBase> FtmwConfig::storage() const
{
    return p_fidStorage;
}

bool FtmwConfig::isComplete() const
{
    if(d_isEnabled)
        return true;

    switch(d_type)
    {
    case Target_Shots:
    case LO_Scan:
    case DR_Scan:
        return completedShots() >= d_targetShots;
        break;
    case Target_Duration:
        return QDateTime::currentDateTime() >= d_targetTime;
        break;
    case Forever:
    case Peak_Up:
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

void FtmwConfig::loadChirps(const int num, const QString path)
{
    ///TODO: Figure out the future of loading chirps from disk
    d_rfConfig.addChirpConfig(ChirpConfig(num,path));
}

void FtmwConfig::loadClocks(const int num, const QString path)
{
    d_rfConfig.loadClockSteps(num,path);
}

void FtmwConfig::prepareToSave()
{
    using namespace BC::Store::FTMW;
    store(enabled,d_isEnabled);
    if(d_isEnabled)
    {
        if(d_type == Target_Duration)
            store(duration,d_duration,"min");
        store(phase,d_phaseCorrectionEnabled);
        store(chirp,d_chirpScoringEnabled);
        if(d_chirpScoringEnabled)
            store(chirpThresh,d_chirpRMSThreshold);
        if(d_chirpScoringEnabled || d_phaseCorrectionEnabled)
            store(chirpOffset,d_chirpOffsetUs,QString::fromUtf8("μs"));
        if(d_targetShots > 0)
            store(tShots,d_targetShots);
    }
}

void FtmwConfig::loadComplete()
{
    using namespace BC::Store::FTMW;
    d_isEnabled = retrieve(enabled,false);
    d_duration = retrieve(duration,0);
    d_phaseCorrectionEnabled = retrieve(phase,false);
    d_chirpScoringEnabled = retrieve(chirp,false);
    d_chirpRMSThreshold = retrieve(chirpThresh,0.0);
    d_chirpOffsetUs = retrieve(chirpOffset,0.0);
    d_type = retrieve(type,Forever);
    d_targetShots = retrieve(tShots,0);
}
