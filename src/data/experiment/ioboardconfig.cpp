#include <src/data/experiment/ioboardconfig.h>

#include <QStringList>
#include <QSettings>
#include <QApplication>

class IOBoardConfigData : public QSharedData
{
public:
    QMap<int,BlackChirp::IOBoardChannel> analog;
    QMap<int,BlackChirp::IOBoardChannel> digital;

    int numAnalog;
    int numDigital;
    int reservedAnalog;
    int reservedDigital;

};

IOBoardConfig::IOBoardConfig(bool fromSettings) : data(new IOBoardConfigData)
{
    if(fromSettings)
    {
        QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
        s.beginGroup(QString("ioboard"));
        s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());

        data->numAnalog = qBound(0,s.value(QString("numAnalog"),4).toInt(),16);
        data->numDigital = qBound(0,s.value(QString("numDigital"),16-data->numAnalog).toInt(),16);
        data->reservedAnalog = qMin(data->numAnalog,s.value(QString("reservedAnalog"),0).toInt());
        data->reservedDigital = qMin(data->numDigital,s.value(QString("reservedDigital"),0).toInt());

        s.endGroup();
        s.endGroup();

        s.beginGroup(QString("iobconfig"));
        s.beginReadArray(QString("analog"));
        for(int i=0; i<data->numAnalog-data->reservedAnalog; i++)
        {
            s.setArrayIndex(i);
            QString name = s.value(QString("name"),QString("ain.%1").arg(i+data->reservedAnalog)).toString();
            bool enabled = s.value(QString("enabled"),false).toBool();
            bool plot = s.value(QString("plot"),false).toBool();
            data->analog.insert(i,BlackChirp::IOBoardChannel(enabled,name,plot));
        }
        s.endArray();
        s.beginReadArray(QString("digital"));
        for(int i=0; i<data->numDigital-data->reservedDigital; i++)
        {
            s.setArrayIndex(i);
            QString name = s.value(QString("name"),QString("din.%1").arg(i+data->reservedDigital)).toString();
            bool enabled = s.value(QString("enabled"),false).toBool();
            bool plot = s.value(QString("plot"),false).toBool();
            data->digital.insert(i,BlackChirp::IOBoardChannel(enabled,name,plot));
        }
        s.endArray();

        s.endGroup();
    }
}

IOBoardConfig::IOBoardConfig(const IOBoardConfig &rhs) : data(rhs.data)
{

}

IOBoardConfig &IOBoardConfig::operator=(const IOBoardConfig &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

IOBoardConfig::~IOBoardConfig()
{

}

void IOBoardConfig::setAnalogChannel(int ch, bool enabled, QString name, bool plot)
{
    if(data->analog.contains(ch))
        data->analog[ch] = BlackChirp::IOBoardChannel(enabled,name,plot);
    else
        data->analog.insert(ch,BlackChirp::IOBoardChannel(enabled,name,plot));
}

void IOBoardConfig::setDigitalChannel(int ch, bool enabled, QString name, bool plot)
{
    if(data->digital.contains(ch))
        data->digital[ch] = BlackChirp::IOBoardChannel(enabled,name,plot);
    else
        data->digital.insert(ch,BlackChirp::IOBoardChannel(enabled,name,plot));
}

void IOBoardConfig::setAnalogChannels(const QMap<int, BlackChirp::IOBoardChannel> l)
{
    data->analog = l;
}

void IOBoardConfig::setDigitalChannels(const QMap<int,BlackChirp::IOBoardChannel> l)
{
    data->digital = l;
}

int IOBoardConfig::numAnalogChannels() const
{
    return data->numAnalog;
}

int IOBoardConfig::numDigitalChannels() const
{
    return data->numDigital;
}

int IOBoardConfig::reservedAnalogChannels() const
{
    return data->reservedAnalog;
}

int IOBoardConfig::reservedDigitalChannels() const
{
    return data->reservedDigital;
}

bool IOBoardConfig::isAnalogChEnabled(int ch) const
{
    if(data->analog.contains(ch))
        return data->analog.value(ch).enabled;
    else
        return false;
}

bool IOBoardConfig::isDigitalChEnabled(int ch) const
{
    if(data->digital.contains(ch))
        return data->digital.value(ch).enabled;
    else
        return false;
}

QString IOBoardConfig::analogChannelName(int ch) const
{
    if(data->analog.contains(ch))
        return data->analog.value(ch).name;
    else
        return QString("");
}

QString IOBoardConfig::digitalChannelName(int ch) const
{
    if(data->digital.contains(ch))
        return data->digital.value(ch).name;
    else
        return QString("");
}

bool IOBoardConfig::plotAnalogChannel(int ch) const
{
    if(data->analog.contains(ch))
        return data->analog.value(ch).plot;
    else
        return false;
}

bool IOBoardConfig::plotDigitalChannel(int ch) const
{
    if(data->digital.contains(ch))
        return data->digital.value(ch).plot;
    else
        return false;
}

QMap<int, BlackChirp::IOBoardChannel> IOBoardConfig::analogList() const
{
    return data->analog;
}

QMap<int, BlackChirp::IOBoardChannel> IOBoardConfig::digitalList() const
{
    return data->digital;
}

QMap<QString, QPair<QVariant, QString> > IOBoardConfig::headerMap() const
{
    QMap<QString, QPair<QVariant, QString> > out;

    auto it = data->analog.constBegin();
    QString prefix = QString("IOBoardConfig");
    QString empty = QString("");
    out.insert(prefix+QString("ReservedAnalog"),qMakePair(data->reservedAnalog,empty));
    out.insert(prefix+QString("ReservedDigital"),qMakePair(data->reservedDigital,empty));
    out.insert(prefix+QString("NumAnalog"),qMakePair(data->numAnalog,empty));
    out.insert(prefix+QString("NumDigital"),qMakePair(data->numDigital,empty));
    for(;it != data->analog.constEnd(); it++)
    {
        out.insert(prefix+QString("Analog.")+QString::number(it.key())+QString(".Enabled"),qMakePair(it.value().enabled,empty));
        out.insert(prefix+QString("Analog.")+QString::number(it.key())+QString(".Name"),qMakePair(it.value().name,empty));
        out.insert(prefix+QString("Analog.")+QString::number(it.key())+QString(".Plot"),qMakePair(it.value().plot,empty));
    }
    it = data->digital.constBegin();
    for(;it != data->digital.constEnd(); it++)
    {
        out.insert(prefix+QString("Digital.")+QString::number(it.key())+QString(".Enabled"),qMakePair(it.value().enabled,empty));
        out.insert(prefix+QString("Digital.")+QString::number(it.key())+QString(".Name"),qMakePair(it.value().name,empty));
        out.insert(prefix+QString("Digital.")+QString::number(it.key())+QString(".Plot"),qMakePair(it.value().plot,empty));
    }

    return out;
}

void IOBoardConfig::parseLine(QString key, QVariant val)
{
    if(key.startsWith(QString("IOBoardConfig")))
    {
        if(key.contains(QString("Reserved")))
        {
            if(key.endsWith(QString("Analog")))
                data->reservedAnalog = val.toInt();
            if(key.endsWith(QString("Digital")))
                data->reservedDigital = val.toInt();
        }
        else if(key.contains(QString("Num")))
        {
            if(key.endsWith(QString("Analog")))
                data->numAnalog = val.toInt();
            if(key.endsWith(QString("Digital")))
                data->numDigital = val.toInt();
        }
        else
        {
            QStringList l = key.split(QString("."));
            if(l.size() < 3)
                return;

            QString subKey = l.constLast();
            int index = l.at(1).toInt();

            QMap<int,BlackChirp::IOBoardChannel> *map = nullptr;
            if(l.constFirst().endsWith(QString("Analog")))
                map = &data->analog;
            else if(l.constFirst().endsWith(QString("Digital")))
                map = &data->digital;

            if(map == nullptr)
                return;

            if(subKey.endsWith(QString("Enabled")))
                map->operator[](index).enabled = val.toBool();
            if(subKey.endsWith(QString("Name")))
                map->operator[](index).name = val.toString();
            if(subKey.endsWith(QString("Plot")))
                map->operator[](index).plot = val.toBool();

        }
    }
}

void IOBoardConfig::saveToSettings() const
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("iobconfig"));
    s.remove(QString("analog"));
    s.beginWriteArray(QString("analog"));
    for(int i=0; i<data->numAnalog-data->reservedAnalog; i++)
    {
        s.setArrayIndex(i);
        QString name = data->analog.value(i).name;
        if(name.isEmpty())
            name = QString("ain.%1").arg(i+data->reservedAnalog);
        s.setValue(QString("name"),name);
        s.setValue(QString("enabled"),data->analog.value(i).enabled);
        s.setValue(QString("plot"),data->analog.value(i).plot);
    }
    s.endArray();
    s.remove(QString("digital"));
    s.beginWriteArray(QString("digital"));
    for(int i=0; i<data->numDigital-data->reservedDigital; i++)
    {
        s.setArrayIndex(i);
        QString name = data->digital.value(i).name;
        if(name.isEmpty())
            name = QString("din.%1").arg(i+data->reservedDigital);
        s.setValue(QString("name"),name);
        s.setValue(QString("enabled"),data->digital.value(i).enabled);
        s.setValue(QString("plot"),data->digital.value(i).plot);
    }
    s.endArray();

    s.endGroup();
    s.sync();
}

