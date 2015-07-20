#include "ioboardconfig.h"

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

IOBoardConfig::IOBoardConfig() : data(new IOBoardConfigData)
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
        QString name = s.value(QString("name"),QString("din.%1").arg(data->reservedDigital)).toString();
        bool enabled = s.value(QString("enabled"),false).toBool();
        bool plot = s.value(QString("plot"),false).toBool();
        data->digital.insert(i,BlackChirp::IOBoardChannel(enabled,name,plot));
    }
    s.endArray();

    s.endGroup();
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
    for(;it != data->analog.constEnd(); it++)
    {
        out.insert(prefix+QString("Analog.")+QString::number(it.key()+data->reservedAnalog)+QString(".Enabled"),qMakePair(it.value().enabled,empty));
        out.insert(prefix+QString("Analog.")+QString::number(it.key()+data->reservedAnalog)+QString(".Name"),qMakePair(it.value().name,empty));
        out.insert(prefix+QString("Analog.")+QString::number(it.key()+data->reservedAnalog)+QString(".Plot"),qMakePair(it.value().plot,empty));
    }
    it = data->digital.constBegin();
    for(;it != data->digital.constEnd(); it++)
    {
        out.insert(prefix+QString("Digital.")+QString::number(it.key()+data->reservedDigital)+QString(".Enabled"),qMakePair(it.value().enabled,empty));
        out.insert(prefix+QString("Digital.")+QString::number(it.key()+data->reservedDigital)+QString(".Name"),qMakePair(it.value().name,empty));
        out.insert(prefix+QString("Digital.")+QString::number(it.key()+data->reservedDigital)+QString(".Plot"),qMakePair(it.value().plot,empty));
    }

    return out;
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

