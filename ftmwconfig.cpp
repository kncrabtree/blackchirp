#include "ftmwconfig.h"

#include <QFile>
#include <QtEndian>

FtmwConfig::FtmwConfig() : data(new FtmwConfigData)
{

}

FtmwConfig::FtmwConfig(const FtmwConfig &rhs) : data(rhs.data)
{

}

FtmwConfig &FtmwConfig::operator=(const FtmwConfig &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

FtmwConfig::~FtmwConfig()
{

}

bool FtmwConfig::isEnabled() const
{
    return data->isEnabled;
}

bool FtmwConfig::isPhaseCorrectionEnabled() const
{
    return data->phaseCorrectionEnabled;
}

bool FtmwConfig::isChirpScoringEnabled() const
{
    return data->chirpScoringEnabled;
}

double FtmwConfig::chirpRMSThreshold() const
{
    return data->chirpRMSThreshold;
}

BlackChirp::FtmwType FtmwConfig::type() const
{
    return data->type;
}

qint64 FtmwConfig::targetShots() const
{
    return data->targetShots;
}

qint64 FtmwConfig::completedShots() const
{
    return data->completedShots;
}

QDateTime FtmwConfig::targetTime() const
{
    return data->targetTime;
}

double FtmwConfig::loFreq() const
{
    return data->loFreq;
}

BlackChirp::Sideband FtmwConfig::sideband() const
{
    return data->sideband;
}

QList<Fid> FtmwConfig::fidList() const
{
    return data->fidList;
}

BlackChirp::FtmwScopeConfig FtmwConfig::scopeConfig() const
{
    return data->scopeConfig;
}

ChirpConfig FtmwConfig::chirpConfig() const
{
    return data->chirpConfig;
}

Fid FtmwConfig::fidTemplate() const
{
    return data->fidTemplate;
}

int FtmwConfig::numFrames() const
{
    return (scopeConfig().summaryFrame && !scopeConfig().manualFrameAverage) ? 1 : scopeConfig().numFrames;
}

QList<Fid> FtmwConfig::parseWaveform(const QByteArray b) const
{

    int np = scopeConfig().recordLength;
    QList<Fid> out;
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


            if(scopeConfig().bytesPerPoint == 1)
            {
                char y = b.at(j*np+i);
                dat = (static_cast<qint64>(y) + static_cast<qint64>(scopeConfig().yOff));
            }
            else if(scopeConfig().bytesPerPoint == 2)
            {
                auto y1 = static_cast<quint8>(b.at(2*(j*np+i)));
                auto y2 = static_cast<quint8>(b.at(2*(j*np+i) + 1));

                qint16 y = 0;
                y |= y1;
                y |= (y2 << 8);

                if(scopeConfig().byteOrder == QDataStream::BigEndian)
                    y = qFromBigEndian(y);
                else
                    y = qFromLittleEndian(y);

                dat = (static_cast<qint64>(y) + static_cast<qint64>(scopeConfig().yOff));
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

                if(scopeConfig().byteOrder == QDataStream::BigEndian)
                    y = qFromBigEndian(y);
                else
                    y = qFromLittleEndian(y);

                dat = (static_cast<qint64>(y) + static_cast<qint64>(scopeConfig().yOff));
            }


            //in peak up mode, add 8 bits of padding so that there are empty bits to fill when
            //the rolling average kicks in.
            //Note that this could lead to overflow problems if bytesPerPoint = 4 and the # of averages is large
            if(type() == BlackChirp::FtmwPeakUp)
                dat = dat << 8;

            if(scopeConfig().blockAverageMultiply)
                dat *= scopeConfig().numAverages;

            d[i] = dat;
        }

        Fid f = fidTemplate();
        f.setData(d);
        f.setShots(scopeConfig().numAverages);

        if(scopeConfig().fastFrameEnabled && scopeConfig().manualFrameAverage)
        {
            if(out.isEmpty())
                out.append(f);
            else
                out[0]+=f;
        }
        else
            out.append(f);
    }

    return out;
}

QVector<qint64> FtmwConfig::extractChirp() const
{
    QVector<qint64> out;
    QList<Fid> dat = fidList();
    if(!dat.isEmpty())
    {
        auto r = chirpRange();
        if(r.first >= 0 && r.second >= 0)
            out = dat.first().rawData().mid(r.first, r.second - r.first);
    }

    return out;
}

QVector<qint64> FtmwConfig::extractChirp(const QByteArray b) const
{
    QVector<qint64> out;
    auto r = chirpRange();
    if(r.first >= 0 && r.second >= 0)
    {
        QList<Fid> l = parseWaveform(b);
        if(!l.isEmpty())
            out = l.first().rawData().mid(r.first, r.second - r.first);
    }

    return out;
}

QString FtmwConfig::errorString() const
{
    return data->errorString;
}

double FtmwConfig::ftMin() const
{
    double sign = 1.0;
    if(data->sideband == BlackChirp::LowerSideband)
        sign = -1.0;
    double lastFreq = data->loFreq + sign*data->scopeConfig.sampleRate/(1e6*2.0);
    return qMin(data->loFreq,lastFreq);
}

double FtmwConfig::ftMax() const
{
    double sign = 1.0;
    if(data->sideband == BlackChirp::LowerSideband)
        sign = -1.0;
    double lastFreq = data->loFreq + sign*data->scopeConfig.sampleRate/(1e6*2.0);
    return qMax(data->loFreq,lastFreq);
}

QPair<int, int> FtmwConfig::chirpRange() const
{
    //want to return [first,last) samples for chirp.
    if(!data->chirpConfig.isValid())
        return qMakePair(-1,-1);

    if(data->fidList.isEmpty())
        return qMakePair(-1,-1);

    //we assume that the scope is triggered at the beginning of the protection pulse

    double chirpStart = (data->chirpConfig.preChirpDelay() + data->chirpConfig.preChirpProtection() - data->scopeConfig.trigDelay*1e6)*1e-6;
    int startSample = qBound(BC_FTMW_MAXSHIFT,qRound(chirpStart*data->scopeConfig.sampleRate) + BC_FTMW_MAXSHIFT,data->fidList.first().size() - BC_FTMW_MAXSHIFT);
    double chirpEnd = chirpStart + data->chirpConfig.chirpDuration(0)*1e-6;
    int endSample = qBound(BC_FTMW_MAXSHIFT,qRound(chirpEnd*data->scopeConfig.sampleRate) - BC_FTMW_MAXSHIFT,data->fidList.first().size() - BC_FTMW_MAXSHIFT);

    if(startSample > endSample)
        qSwap(startSample,endSample);

    return qMakePair(startSample,endSample);
}

bool FtmwConfig::writeFidFile(int num, int snapNum) const
{
    QFile fid(BlackChirp::getExptFile(num,BlackChirp::FidFile,QString(""),snapNum));
    if(fid.open(QIODevice::WriteOnly))
    {
        QDataStream d(&fid);
        d << Fid::magicString();
        d << data->fidList;
        fid.close();
        return true;
    }
    else
        return false;
}

bool FtmwConfig::writeFidFile(int num, QList<Fid> list, QString path)
{
    QFile fid(BlackChirp::getExptFile(num,BlackChirp::FidFile,path));
    if(fid.open(QIODevice::WriteOnly))
    {
        QDataStream d(&fid);
        d << Fid::magicString();
        d << list;
        fid.close();
        return true;
    }
    else
        return false;
}

bool FtmwConfig::prepareForAcquisition()
{
    Fid f(scopeConfig().xIncr,loFreq(),QVector<qint64>(0),sideband(),scopeConfig().yMult,1);

    //in peak up mode, data points will be shifted by 8 bits (x256), so the multiplier
    //needs to decrease by a factor of 256
    if(type() == BlackChirp::FtmwPeakUp)
        f.setVMult(f.vMult()/256.0);
    data->fidTemplate = f;

    if(!chirpConfig().isValid())
    {
        data->errorString = QString("Invalid chirp configuration.");
        return false;
    }

    return true;


}

void FtmwConfig::setEnabled()
{
    data->isEnabled = true;
}

void FtmwConfig::setPhaseCorrectionEnabled(bool enabled)
{
    data->phaseCorrectionEnabled = enabled;
}

void FtmwConfig::setChirpScoringEnabled(bool enabled)
{
    data->chirpScoringEnabled = enabled;
}

void FtmwConfig::setChirpRMSThreshold(double t)
{
    data->chirpRMSThreshold = t;
}

void FtmwConfig::setFidTemplate(const Fid f)
{
    data->fidTemplate = f;
}

void FtmwConfig::setType(const BlackChirp::FtmwType type)
{
    data->type = type;
}

void FtmwConfig::setTargetShots(const qint64 target)
{
    data->targetShots = target;
}

void FtmwConfig::increment()
{
    int increment = scopeConfig().numAverages;
    if(scopeConfig().fastFrameEnabled && (scopeConfig().manualFrameAverage || scopeConfig().summaryFrame))
        increment *= scopeConfig().numFrames;

    if(type() == BlackChirp::FtmwPeakUp)
        data->completedShots = qMin(completedShots()+increment,targetShots());
    else
        data->completedShots+=increment;
}

void FtmwConfig::setTargetTime(const QDateTime time)
{
    data->targetTime = time;
}

void FtmwConfig::setLoFreq(const double f)
{
    data->loFreq = f;
}

void FtmwConfig::setSideband(const BlackChirp::Sideband sb)
{
    data->sideband = sb;
}

bool FtmwConfig::setFidsData(const QList<QVector<qint64> > newList)
{
    if(data->fidList.isEmpty())
    {
        for(int i=0; i<newList.size(); i++)
        {
            Fid f = fidTemplate();
            f.setData(newList.at(i));
            f.setShots(scopeConfig().numAverages);
            data->fidList.append(f);
        }
    }
    else
    {
        if(newList.size() != data->fidList.size())
        {
            data->errorString = QString("Could not set new FID list data. List sizes are not equal (new = %1, current = %2)")
                    .arg(newList.size()).arg(data->fidList.size());
            return false;
        }

        for(int i=0; i<data->fidList.size(); i++)
        {
            data->fidList[i].setData(newList.at(i));
            if(type() == BlackChirp::FtmwPeakUp)
                data->fidList[i].setShots(qMin(completedShots()+scopeConfig().numAverages,targetShots()));
            else
                data->fidList[i].setShots(completedShots()+scopeConfig().numAverages);
        }
    }

    return true;
}

bool FtmwConfig::addFids(const QByteArray rawData, int shift)
{
    QList<Fid> newList = parseWaveform(rawData);
    if(data->completedShots > 0)
    {
        if(newList.size() != data->fidList.size())
        {
            data->errorString = QString("Could not set new FID list data. List sizes are not equal (new = %1, current = %2)")
                    .arg(newList.size()).arg(data->fidList.size());
            return false;
        }

        if(type() == BlackChirp::FtmwPeakUp)
        {
            if(targetShots() > 1)
            {
                for(int i=0; i<data->fidList.size(); i++)
                    newList[i].rollingAverage(data->fidList.at(i),targetShots(),shift);
            }
            else
                data->fidList = newList;
        }
        else
        {
            for(int i=0; i<data->fidList.size(); i++)
                newList[i].add(data->fidList.at(i),shift);
        }
    }
    data->fidList = newList;

    return true;
}

bool FtmwConfig::subtractFids(const QList<Fid> otherList)
{
    if(otherList.size() != data->fidList.size())
        return false;

    for(int i=0; i<otherList.size(); i++)
    {
        if(otherList.at(i).size() != data->fidList.at(i).size())
            return false;

        if(otherList.at(i).shots() >= data->fidList.at(i).shots())
            return false;
    }

    for(int i=0; i<data->fidList.size(); i++)
        data->fidList[i] -= otherList.at(i);

    return true;
}

void FtmwConfig::resetFids()
{
    data->fidList.clear();
    data->completedShots = 0;
}

void FtmwConfig::setScopeConfig(const BlackChirp::FtmwScopeConfig &other)
{
    data->scopeConfig = other;
}

void FtmwConfig::setChirpConfig(const ChirpConfig other)
{
    data->chirpConfig = other;
}

bool FtmwConfig::isComplete() const
{
    if(!isEnabled())
        return true;

    switch(type())
    {
    case BlackChirp::FtmwTargetShots:
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
    if(data->fidList.isEmpty())
        out.insert(prefix+QString("CompletedShots"),qMakePair(0,empty));
    else
        out.insert(prefix+QString("CompletedShots"),qMakePair(data->fidList.first().shots(),empty));
    out.insert(prefix+QString("LoFrequency"),qMakePair(QString::number(loFreq(),'f',6),QString("MHz")));
    out.insert(prefix+QString("Sideband"),qMakePair((int)sideband(),empty));
    out.insert(prefix+QString("FidVMult"),qMakePair(QString::number(fidTemplate().vMult(),'g',12),QString("V")));
    out.insert(prefix+QString("PhaseCorrection"),qMakePair(data->phaseCorrectionEnabled,QString("")));
    out.insert(prefix+QString("ChirpScoring"),qMakePair(data->chirpScoringEnabled,QString("")));
    out.insert(prefix+QString("ChirpRMSThreshold"),qMakePair(QString::number(data->chirpRMSThreshold,'f',3),QString("")));


    out.unite(data->scopeConfig.headerMap());
    out.unite(data->chirpConfig.headerMap());

    return out;

}

void FtmwConfig::loadFids(const int num, const QString path)
{
    QFile fid(BlackChirp::getExptFile(num,BlackChirp::FidFile,path));
    if(fid.open(QIODevice::ReadOnly))
    {
        QDataStream d(&fid);
        QByteArray magic;
        d >> magic;
        if(magic.startsWith("BCFID"))
        {
            if(magic.endsWith("v1.0"))
            {
                QList<Fid> dat;
                d >> dat;
                data->fidList = dat;
                if(!dat.isEmpty())
                    data->fidTemplate = dat.first();
                data->fidTemplate.setData(QVector<qint64>());
            }
        }
        fid.close();
    }

    if(data->fidList.isEmpty())
    {
        //try to reconstruct from snapshots, if any
        QFile snp(BlackChirp::getExptFile(num,BlackChirp::SnapFile,path));
        if(snp.exists() && snp.open(QIODevice::ReadOnly))
        {
            bool parseSuccess = false;
            int numSnaps = 0;
            while(!snp.atEnd())
            {
                QString line = snp.readLine();
                if(line.startsWith(QString("fid")))
                {
                    QStringList l = line.split(QString("\t"));
                    bool ok = false;
                    int n = l.last().trimmed().toInt(&ok);
                    if(ok)
                    {
                        parseSuccess = true;
                        numSnaps = n;
                        break;
                    }
                }
            }

            if(parseSuccess && numSnaps > 0)
            {
                for(int i=0; i<numSnaps; i++)
                {
                    QList<Fid> in;
                    QFile f(BlackChirp::getExptFile(num,BlackChirp::FidFile,path,i));
                    if(f.exists() && f.open(QIODevice::ReadOnly))
                    {
                        QDataStream d(&f);
                        QByteArray magic;
                        d >> magic;
                        if(magic.startsWith("BCFID"))
                        {
                            if(magic.endsWith("v1.0"))
                                d >> in;
                        }
                        f.close();
                        if(data->fidList.isEmpty())
                            data->fidList = in;
                        else if(data->fidList.size() == in.size())
                        {
                            for(int j=0; j<data->fidList.size(); j++)
                                data->fidList[j] += in.at(j);
                        }
                    }
                }
            }
        }
    }
}

void FtmwConfig::parseLine(const QString key, const QVariant val)
{
    if(key.startsWith(QString("FtmwScope")))
    {
        if(key.endsWith(QString("FidChannel")))
            data->scopeConfig.fidChannel = val.toInt();
        if(key.endsWith(QString("VerticalScale")))
            data->scopeConfig.vScale = val.toDouble();
        if(key.endsWith(QString("VerticalOffset")))
            data->scopeConfig.vOffset = val.toDouble();
        if(key.endsWith(QString("TriggerChannel")))
        {
            if(val.toString().contains(QString("AuxIn")))
                data->scopeConfig.trigChannel = 0;
            else
                data->scopeConfig.trigChannel = val.toInt();
        }
        if(key.endsWith(QString("TriggerDelay")))
            data->scopeConfig.trigDelay = val.toDouble();
        if(key.endsWith(QString("TriggerLevel")))
            data->scopeConfig.trigLevel = val.toDouble();
        if(key.endsWith(QString("TriggerSlope")))
        {
            if(val.toString().contains(QString("Rising")))
                data->scopeConfig.slope = BlackChirp::RisingEdge;
            else
                data->scopeConfig.slope = BlackChirp::FallingEdge;
        }
        if(key.endsWith(QString("SampleRate")))
            data->scopeConfig.sampleRate = val.toDouble()*1e9;
        if(key.endsWith(QString("RecordLength")))
            data->scopeConfig.recordLength = val.toInt();
        if(key.endsWith(QString("FastFrame")))
            data->scopeConfig.fastFrameEnabled = val.toBool();
        if(key.endsWith(QString("SummaryFrame")))
            data->scopeConfig.summaryFrame = val.toBool();
        if(key.endsWith(QString("BlockAverage")))
            data->scopeConfig.blockAverageEnabled = val.toBool();
        if(key.endsWith(QString("NumAverages")))
            data->scopeConfig.numAverages = val.toInt();
        if(key.endsWith(QString("BytesPerPoint")))
            data->scopeConfig.bytesPerPoint = val.toInt();
        if(key.endsWith(QString("NumFrames")))
            data->scopeConfig.numFrames = val.toInt();
        if(key.endsWith(QString("ByteOrder")))
        {
            if(val.toString().contains(QString("BigEndian")))
                data->scopeConfig.byteOrder = QDataStream::BigEndian;
            else
                data->scopeConfig.byteOrder = QDataStream::LittleEndian;
        }
    }
    else if(key.startsWith(QString("FtmwConfig")))
    {
        if(key.endsWith(QString("Enabled")))
            data->isEnabled = val.toBool();
        if(key.endsWith(QString("Type")))
            data->type = (BlackChirp::FtmwType)val.toInt();
        if(key.endsWith(QString("TargetShots")))
            data->targetShots = val.toInt();
        if(key.endsWith(QString("TargetTime")))
            data->targetTime = val.toDateTime();
        if(key.endsWith(QString("LoFrequency")))
            data->loFreq = val.toDouble();
        if(key.endsWith(QString("Sideband")))
            data->sideband = (BlackChirp::Sideband)val.toInt();
        if(key.endsWith(QString("PhaseCorrection")))
            data->phaseCorrectionEnabled = val.toBool();
        if(key.endsWith(QString("ChirpScoring")))
            data->chirpScoringEnabled = val.toBool();
        if(key.endsWith(QString("ChirpRMSThreshold")))
            data->chirpRMSThreshold = val.toDouble();
    }
}

void FtmwConfig::loadChirps(const int num, const QString path)
{
    data->chirpConfig = ChirpConfig(num,path);
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
    s.setValue(QString("chirpRMSThreshold"),chirpRMSThreshold());

    s.setValue(QString("fidChannel"),scopeConfig().fidChannel);
    s.setValue(QString("vScale"),scopeConfig().vScale);
    s.setValue(QString("triggerChannel"),scopeConfig().trigChannel);
    s.setValue(QString("triggerDelay"),scopeConfig().trigDelay);
    s.setValue(QString("triggerLevel"),scopeConfig().trigLevel);
    s.setValue(QString("triggerSlope"),static_cast<int>(scopeConfig().slope));
    s.setValue(QString("sampleRate"),scopeConfig().sampleRate);
    s.setValue(QString("recordLength"),scopeConfig().recordLength);
    s.setValue(QString("bytesPerPoint"),scopeConfig().bytesPerPoint);
    s.setValue(QString("fastFrame"),scopeConfig().fastFrameEnabled);
    s.setValue(QString("numFrames"),scopeConfig().numFrames);
    s.setValue(QString("summaryFrame"),scopeConfig().summaryFrame);
    s.setValue(QString("blockAverage"),scopeConfig().blockAverageEnabled);
    s.setValue(QString("numAverages"),scopeConfig().numAverages);
    s.setValue(QString("loFreq"),loFreq());
    s.setValue(QString("sideband"),static_cast<int>(sideband()));

    s.endGroup();

    chirpConfig().saveToSettings();


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
    out.setScopeConfig(sc);

    out.setLoFreq(s.value(QString("loFreq"),0.0).toDouble());
    out.setSideband(static_cast<BlackChirp::Sideband>(s.value(QString("sideband"),BlackChirp::UpperSideband).toInt()));

    out.setChirpConfig(ChirpConfig::loadFromSettings());

    return out;
}

