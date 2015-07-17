#include "ioboardconfig.h"

#include <QStringList>
#include <QSettings>
#include <QApplication>

class IOBoardConfigData : public QSharedData
{
public:
    QMap<int,QPair<bool,QString>> analog;
    QMap<int,QPair<bool,QString>> digital;

};

IOBoardConfig::IOBoardConfig() : data(new IOBoardConfigData)
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("ioboard"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());

    int numAnalog = qBound(0,s.value(QString("numAnalog"),4).toInt(),16);
    int numDigital = qBound(0,s.value(QString("numDigital"),16-numAnalog).toInt(),16);
    int reservedAnalog = qMin(numAnalog,s.value(QString("reservedAnalog"),0).toInt());
    int reservedDigital = qMin(numDigital,s.value(QString("reservedDigital"),0).toInt());

    s.endGroup();
    s.endGroup();

    s.beginGroup(QString("iobconfig"));
    s.beginReadArray(QString("analog"));
    for(int i=reservedAnalog; i<numAnalog; i++)
    {
        s.setArrayIndex(i);
        QString name = s.value(QString("name"),QString("")).toString();
        bool enabled = s.value(QString("enabled"),false).toBool();
        data->analog.insert(i,qMakePair(enabled,name));
    }
    s.endArray();
    s.beginReadArray(QString("digital"));
    for(int i=reservedDigital; i<numDigital; i++)
    {
        s.setArrayIndex(i);
        QString name = s.value(QString("name"),QString("")).toString();
        bool enabled = s.value(QString("enabled"),false).toBool();
        data->digital.insert(i,qMakePair(enabled,name));
    }

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

void IOBoardConfig::setAnalogChannel(int ch, bool enabled, QString name)
{
    if(data->analog.contains(ch))
        data->analog[ch] = qMakePair(enabled,name);
    else
        data->analog.insert(ch,qMakePair(enabled,name));
}

void IOBoardConfig::setDigitalChannel(int ch, bool enabled, QString name)
{
    if(data->digital.contains(ch))
        data->digital[ch] = qMakePair(enabled,name);
    else
        data->digital.insert(ch,qMakePair(enabled,name));
}

void IOBoardConfig::setAnalogChannels(const QMap<int, QPair<bool, QString> > l)
{
    data->analog = l;
}

void IOBoardConfig::setDigitalChannels(const QMap<int, QPair<bool, QString> > l)
{
    data->digital = l;
}

bool IOBoardConfig::isAnalogChEnabled(int ch) const
{
    if(data->analog.contains(ch))
        return data->analog.value(ch).first;
    else
        return false;
}

bool IOBoardConfig::isDigitalChEnabled(int ch) const
{
    if(data->digital.contains(ch))
        return data->digital.value(ch).first;
    else
        return false;
}

QMap<int, QPair<bool, QString> > IOBoardConfig::analogList() const
{
    return data->analog;
}

QMap<int, QPair<bool, QString> > IOBoardConfig::digitalList() const
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
        out.insert(prefix+QString("Analog.")+QString::number(it.key())+QString(".Enabled"),qMakePair(it.value().first,empty));
        out.insert(prefix+QString("Analog.")+QString::number(it.key())+QString(".Name"),qMakePair(it.value().second,empty));
    }
    it = data->digital.constBegin();
    for(;it != data->digital.constEnd(); it++)
    {
        out.insert(prefix+QString("Digital.")+QString::number(it.key())+QString(".Enabled"),qMakePair(it.value().first,empty));
        out.insert(prefix+QString("Digital.")+QString::number(it.key())+QString(".Name"),qMakePair(it.value().second,empty));
    }

    return out;
}

