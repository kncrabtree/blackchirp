#ifndef FTMWCONFIG_H
#define FTMWCONFIG_H

#include <QSharedDataPointer>

#include <QDateTime>
#include <QDataStream>
#include <QVariant>
#include <QMetaType>

#include "fid.h"
#include "rfconfig.h"
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
    bool isChirpScoringEnabled() const;
    double chirpRMSThreshold() const;
    double chirpOffsetUs() const;
    BlackChirp::FtmwType type() const;
    qint64 targetShots() const;
    qint64 completedShots() const;
    QDateTime targetTime() const;
    Fid singleFid(int frame=0, int segment=0) const;
    FidList fidList() const;
    FidList fidList(int segment) const;
    QVector<qint64> rawFidList() const;
    QList<FidList> multiFidList() const;
    BlackChirp::FtmwScopeConfig scopeConfig() const;
    RfConfig rfConfig() const;
    ChirpConfig chirpConfig(int num = 0) const;
    Fid fidTemplate() const;
    bool processingPaused() const;
    int numFrames() const;
    FidList parseWaveform(const QByteArray b) const;
    QVector<qint64> extractChirp() const;
    QVector<qint64> extractChirp(const QByteArray b) const;
    QString errorString() const;
    double ftMin() const;
    double ftMax() const;
    QPair<int,int> chirpRange() const;
    bool writeFids(int num, int snapNum = -1) const;
    static bool writeFidFile(int num, FidList list, QString path = QString(""));

    bool prepareForAcquisition();
    void setEnabled(bool en = true);
    void setPhaseCorrectionEnabled(bool enabled);
    void setChirpScoringEnabled(bool enabled);
    void setChirpRMSThreshold(double t);
    void setChirpOffsetUs(double o);
    void setFidTemplate(const Fid f);
    void setType(const BlackChirp::FtmwType type);
    void setTargetShots(const qint64 target);
    bool increment();
    void setTargetTime(const QDateTime time);
    bool setFidsData(const QList<QVector<qint64>> newList);
    bool addFids(const QByteArray rawData, int shift = 0);
    bool subtractFids(const FtmwConfig other);
    void resetFids();
    void setScopeConfig(const BlackChirp::FtmwScopeConfig &other);
    void setRfConfig(const RfConfig other);
    void clocksReady();


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
    FtmwConfigData() : isEnabled(false), phaseCorrectionEnabled(false), chirpScoringEnabled(false), chirpRMSThreshold(0.0),
        chirpOffsetUs(-1.0), type(BlackChirp::FtmwForever), targetShots(-1), completedShots(0), multipleFidLists(false),
        processingPaused(false) {}

    bool isEnabled;
    bool phaseCorrectionEnabled;
    bool chirpScoringEnabled;
    double chirpRMSThreshold;
    double chirpOffsetUs;
    BlackChirp::FtmwType type;
    qint64 targetShots;
    qint64 completedShots;
    QDateTime targetTime;

    bool multipleFidLists;
    FidList fidList;
    QList<FidList> multiFidStorage;

    BlackChirp::FtmwScopeConfig scopeConfig;
    RfConfig rfConfig;
    Fid fidTemplate;
    QString errorString;

    bool processingPaused;

};



#endif // FTMWCONFIG_H
