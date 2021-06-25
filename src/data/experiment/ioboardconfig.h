#ifndef IOBOARDCONFIG_H
#define IOBOARDCONFIG_H

#include <QSharedDataPointer>

#include <QMap>
#include <QVariant>

#include <data/datastructs.h>

class IOBoardConfigData;

class IOBoardConfig
{
public:
    IOBoardConfig(bool fromSettings = true);
    IOBoardConfig(const IOBoardConfig &);
    IOBoardConfig &operator=(const IOBoardConfig &);
    ~IOBoardConfig();

    void setAnalogChannel(int ch, bool enabled, QString name, bool plot);
    void setDigitalChannel(int ch, bool enabled, QString name, bool plot);
    void setAnalogChannels(const  QMap<int,BlackChirp::IOBoardChannel> l);
    void setDigitalChannels(const QMap<int, BlackChirp::IOBoardChannel> l);

    int numAnalogChannels() const;
    int numDigitalChannels() const;
    int reservedAnalogChannels() const;
    int reservedDigitalChannels() const;
    bool isAnalogChEnabled(int ch) const;
    bool isDigitalChEnabled(int ch) const;
    QString analogChannelName(int ch) const;
    QString digitalChannelName(int ch) const;
    bool plotAnalogChannel(int ch) const;
    bool plotDigitalChannel(int ch) const;
    QMap<int,BlackChirp::IOBoardChannel> analogList() const;
    QMap<int,BlackChirp::IOBoardChannel> digitalList() const;

    QMap<QString,QPair<QVariant,QString>> headerMap() const;
    void parseLine(QString key, QVariant val);
    void saveToSettings() const;

private:
    QSharedDataPointer<IOBoardConfigData> data;
};

#endif // IOBOARDCONFIG_H
