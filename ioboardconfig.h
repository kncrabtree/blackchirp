#ifndef IOBOARDCONFIG_H
#define IOBOARDCONFIG_H

#include <QSharedDataPointer>

#include <QMap>
#include <QVariant>

class IOBoardConfigData;

class IOBoardConfig
{
public:
    IOBoardConfig();
    IOBoardConfig(const IOBoardConfig &);
    IOBoardConfig &operator=(const IOBoardConfig &);
    ~IOBoardConfig();

    void setAnalogChannel(int ch, bool enabled, QString name);
    void setDigitalChannel(int ch, bool enabled, QString name);
    void setAnalogChannels(const  QMap<int,QPair<bool,QString>> l);
    void setDigitalChannels(const  QMap<int,QPair<bool,QString>> l);

    bool isAnalogChEnabled(int ch) const;
    bool isDigitalChEnabled(int ch) const;
    QMap<int,QPair<bool,QString>> analogList() const;
    QMap<int,QPair<bool,QString>> digitalList() const;

    QMap<QString,QPair<QVariant,QString>> headerMap() const;

private:
    QSharedDataPointer<IOBoardConfigData> data;
};

#endif // IOBOARDCONFIG_H
