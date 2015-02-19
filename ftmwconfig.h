#ifndef FTMWCONFIG_H
#define FTMWCONFIG_H

#include <QSharedDataPointer>
#include "oscilloscope.h"
#include <QDateTime>
#include "fid.h"

class FtmwConfigData;

class FtmwConfig
{
public:
    FtmwConfig();
    FtmwConfig(const FtmwConfig &);
    FtmwConfig &operator=(const FtmwConfig &);
    ~FtmwConfig();

    enum FtmwType
    {
        TargetShots,
        TargetTime,
        Forever,
        PeakUp
    };

    bool isEnabled() const;
    FtmwConfig::FtmwType type() const;
    qint64 targetShots() const;
    qint64 completedShots() const;
    QDateTime targetTime() const;
    int autoSaveShots() const;
    double loFreq() const;
    Fid::Sideband sideband() const;
    QList<Fid> fidList() const;
    Oscilloscope::ScopeConfig scopeConfig() const;

    void setEnabled();
    void setType(const FtmwConfig::FtmwType type);
    void setTargetShots(const qint64 target);
    void increment();
    void setTargetTime(const QDateTime time);
    void setAutoSaveShots(const int shots);
    void setLoFreq(const double f);
    void setSideband(const Fid::Sideband sb);
    void setFidList(const QList<Fid> list);
    void setScopeConfig(const Oscilloscope::ScopeConfig &other);

    bool isComplete() const;
    QHash<QString,QPair<QVariant,QString> > headerHash() const;

private:
    QSharedDataPointer<FtmwConfigData> data;
};

#endif // FTMWCONFIG_H
