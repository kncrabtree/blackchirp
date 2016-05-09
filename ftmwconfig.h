#ifndef FTMWCONFIG_H
#define FTMWCONFIG_H

#include <QSharedDataPointer>

#include <QDateTime>
#include <QDataStream>
#include <QVariant>
#include <QMetaType>

#include "fid.h"
#include "chirpconfig.h"
#include "datastructs.h"

#define BC_FTMW_MAXSHIFT 50

class FtmwConfigData;

class FtmwConfig
{
public:
    FtmwConfig();
    FtmwConfig(const FtmwConfig &);
    FtmwConfig &operator=(const FtmwConfig &);
    ~FtmwConfig();

    bool isEnabled() const;
    bool isPhaseCorrectionEnabled() const;
    BlackChirp::FtmwType type() const;
    qint64 targetShots() const;
    qint64 completedShots() const;
    QDateTime targetTime() const;
    double loFreq() const;
    BlackChirp::Sideband sideband() const;
    QList<Fid> fidList() const;
    BlackChirp::FtmwScopeConfig scopeConfig() const;
    ChirpConfig chirpConfig() const;
    Fid fidTemplate() const;
    int numFrames() const;
    QList<Fid> parseWaveform(QByteArray b) const;
    QVector<qint64> extractChirp(QByteArray b) const;
    QString errorString() const;
    double ftMin() const;
    double ftMax() const;
    QPair<int,int> chirpRange() const;
    bool writeFidFile(int num, int snapNum = -1) const;
    static bool writeFidFile(int num, QList<Fid> list, QString path = QString(""));

    bool prepareForAcquisition();
    void setEnabled();
    void setPhaseCorrectionEnabled(bool enabled);
    void setFidTemplate(const Fid f);
    void setType(const BlackChirp::FtmwType type);
    void setTargetShots(const qint64 target);
    void increment();
    void setTargetTime(const QDateTime time);
    void setLoFreq(const double f);
    void setSideband(const BlackChirp::Sideband sb);
    bool setFidsData(const QList<QVector<qint64>> newList);
    bool addFids(const QByteArray rawData, int shift = 0);
    bool subtractFids(const QList<Fid> otherList);
    void resetFids();
    void setScopeConfig(const BlackChirp::FtmwScopeConfig &other);
    void setChirpConfig(const ChirpConfig other);


    bool isComplete() const;
    QMap<QString,QPair<QVariant,QString> > headerMap() const;
    void loadFids(const int num, const QString path = QString(""));
    void parseLine(const QString key, const QVariant val);
    void loadChirps(const int num, const QString path = QString(""));

    void saveToSettings() const;
    static FtmwConfig loadFromSettings();

private:
    QSharedDataPointer<FtmwConfigData> data;

};

class FtmwConfigData : public QSharedData
{
public:
    FtmwConfigData() : isEnabled(false), phaseCorrectionEnabled(false), type(BlackChirp::FtmwForever), targetShots(-1),
        completedShots(0), loFreq(0.0), sideband(BlackChirp::UpperSideband) {}

    bool isEnabled;
    bool phaseCorrectionEnabled;
    BlackChirp::FtmwType type;
    qint64 targetShots;
    qint64 completedShots;
    QDateTime targetTime;

    double loFreq;
    BlackChirp::Sideband sideband;
    QList<Fid> fidList;

    BlackChirp::FtmwScopeConfig scopeConfig;
    ChirpConfig chirpConfig;
    Fid fidTemplate;
    QString errorString;

};



#endif // FTMWCONFIG_H
