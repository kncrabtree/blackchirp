#include "ftmwconfig.h"

#include <QFile>

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
    return scopeConfig().summaryFrame ? 1 : scopeConfig().numFrames;
}

QList<Fid> FtmwConfig::parseWaveform(QByteArray b) const
{

    int np = scopeConfig().recordLength;
    QList<Fid> out;
    //read raw data into vector in 64 bit integer form
    for(int j=0;j<numFrames();j++)
    {
        QVector<qint64> d(np);

        for(int i=0; i<np;i++)
        {
            if(scopeConfig().bytesPerPoint == 1)
            {
                char y = b.at(j*np+i);
                d[i] = (static_cast<qint64>(y) + static_cast<qint64>(scopeConfig().yOff));
            }
            else
            {
                unsigned char y1 = b.at(2*(j*np+i));
                unsigned char y2 = b.at(2*(j*np+i) + 1);
                qint16 y = 0;
                if(scopeConfig().byteOrder == QDataStream::LittleEndian)
                {
                    y |= static_cast<quint8>(y1);
                    y |= static_cast<quint8>(y2) << 8;
                }
                else
                {
                    y |= static_cast<quint8>(y1) << 8;
                    y |= static_cast<quint8>(y2);
                }
                d[i] = (static_cast<qint64>(y) + static_cast<qint64>(scopeConfig().yOff));
            }
        }

        Fid f = fidTemplate();
        f.setData(d);
        out.append(f);
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

    double chirpStart = (data->chirpConfig.preChirpDelay() + data->chirpConfig.preChirpProtection() - data->scopeConfig.trigDelay)*1e-6;
    int startSample = qBound(BC_FTMW_MAXSHIFT,qRound(chirpStart*data->scopeConfig.sampleRate) + BC_FTMW_MAXSHIFT,data->fidList.first().size() - BC_FTMW_MAXSHIFT);
    double chirpEnd = chirpStart + data->chirpConfig.chirpDuration()*1e-6;
    int endSample = qBound(BC_FTMW_MAXSHIFT,qRound(chirpEnd*data->scopeConfig.sampleRate) - BC_FTMW_MAXSHIFT,data->fidList.first().size() - BC_FTMW_MAXSHIFT);

    if(startSample > endSample)
        qSwap(startSample,endSample);

    return qMakePair(startSample,endSample);
}

bool FtmwConfig::writeFidFile(int num, int snapNum) const
{
    QFile fid(BlackChirp::getExptFile(num,BlackChirp::FidFile,snapNum));
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

bool FtmwConfig::writeFidFile(int num, QList<Fid> list)
{
    QFile fid(BlackChirp::getExptFile(num,BlackChirp::FidFile));
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
    if(type() == BlackChirp::FtmwPeakUp)
        data->completedShots = qMin(completedShots()+1,targetShots());
    else
        data->completedShots++;
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
                data->fidList[i].setShots(qMin(completedShots()+1,targetShots()));
            else
                data->fidList[i].setShots(completedShots()+1);
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
            for(int i=0; i<data->fidList.size(); i++)
                newList[i].rollingAverage(data->fidList.at(i),targetShots());
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
    out.insert(prefix+QString("LoFrequency"),qMakePair(QString::number(loFreq(),'f',6),QString("MHz")));
    out.insert(prefix+QString("Sideband"),qMakePair((int)sideband(),empty));
    out.insert(prefix+QString("FidVMult"),qMakePair(QString::number(fidTemplate().vMult(),'g',12),QString("V")));
    out.insert(prefix+QString("PhaseCorrection"),qMakePair(data->phaseCorrectionEnabled,QString("")));


    out.unite(data->scopeConfig.headerMap());
    out.unite(data->chirpConfig.headerMap());

    return out;

}

void FtmwConfig::loadFids(const int num)
{
    QFile fid(BlackChirp::getExptFile(num,BlackChirp::FidFile));
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
            }
        }
        fid.close();
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
        if(key.endsWith(QString("BytesPerPoint")))
            data->scopeConfig.bytesPerPoint = val.toInt();
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
    }
}

void FtmwConfig::loadChirps(const int num)
{
    data->chirpConfig = ChirpConfig(num);
}

