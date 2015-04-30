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

#ifdef BC_CUDA
#include "gpuaverager.h"
#endif

class FtmwConfigData;

class FtmwConfig
{
public:
    FtmwConfig();
    FtmwConfig(const FtmwConfig &);
    FtmwConfig &operator=(const FtmwConfig &);
    ~FtmwConfig();

    bool isEnabled() const;
    BlackChirp::FtmwType type() const;
    qint64 targetShots() const;
    qint64 completedShots() const;
    QDateTime targetTime() const;
    int autoSaveShots() const;
    double loFreq() const;
    BlackChirp::Sideband sideband() const;
    QList<Fid> fidList() const;
    BlackChirp::FtmwScopeConfig scopeConfig() const;
    ChirpConfig chirpConfig() const;
    Fid fidTemplate() const;
    int numFrames() const;
    QList<Fid> parseWaveform(QByteArray b) const;
    QString errorString() const;

    bool prepareForAcquisition();
    void setEnabled();
    void setFidTemplate(const Fid f);
    void setType(const BlackChirp::FtmwType type);
    void setTargetShots(const qint64 target);
    void increment();
    void setTargetTime(const QDateTime time);
    void setAutoSaveShots(const int shots);
    void setLoFreq(const double f);
    void setSideband(const BlackChirp::Sideband sb);
    bool setFids(const QByteArray newData);
    bool addFids(const QByteArray rawData);
    void resetFids();
    void setScopeConfig(const BlackChirp::FtmwScopeConfig &other);
    void setChirpConfig(const ChirpConfig other);


    bool isComplete() const;
    QMap<QString,QPair<QVariant,QString> > headerMap() const;

private:
    QSharedDataPointer<FtmwConfigData> data;

};

class FtmwConfigData : public QSharedData
{
public:
    FtmwConfigData() : isEnabled(false), type(BlackChirp::FtmwForever), targetShots(-1), completedShots(0), autoSaveShots(1000), loFreq(0.0), sideband(BlackChirp::UpperSideband) {}

    bool isEnabled;
    BlackChirp::FtmwType type;
    qint64 targetShots;
    qint64 completedShots;
    QDateTime targetTime;
    int autoSaveShots;

    double loFreq;
    BlackChirp::Sideband sideband;
    QList<Fid> fidList;

    BlackChirp::FtmwScopeConfig scopeConfig;
    ChirpConfig chirpConfig;
    Fid fidTemplate;
    QString errorString;

#ifdef BC_CUDA
    GpuAverager gpuAvg;
#endif

};


Q_DECLARE_TYPEINFO(FtmwConfig, Q_MOVABLE_TYPE);


#endif // FTMWCONFIG_H
