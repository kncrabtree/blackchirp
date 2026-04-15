#ifndef FTMWCONFIGTYPES_H
#define FTMWCONFIGTYPES_H

#include <data/experiment/ftmwconfig.h>
#include <QDateTime>

class FtmwConfigSingle : public FtmwConfig
{
public:
    FtmwConfigSingle(const QString& scopeHwKey);
    FtmwConfigSingle(const FtmwConfig &other);
    ~FtmwConfigSingle() {}

    // ExperimentObjective interface
    int perMilComplete() const override;
    bool isComplete() const override;

    // FtmwConfig interface
    quint64 completedShots() const override;
protected:
    bool _init() override;
    void _prepareToSave() override;
    void _loadComplete() override;
    std::shared_ptr<FidStorageBase> createStorage(int num, QString path="") override;
};


class FtmwConfigPeakUp : public FtmwConfig
{
public:
    FtmwConfigPeakUp(const QString& scopeHwKey);
    FtmwConfigPeakUp(const FtmwConfig &other);
    ~FtmwConfigPeakUp() {}

    // ExperimentObjective interface
    int perMilComplete() const override;
    bool isComplete() const override;

    // FtmwConfig interface
    quint64 completedShots() const override;
protected:
    quint8 bitShift() const override;
    bool _init() override;
    void _prepareToSave() override;
    void _loadComplete() override;
    std::shared_ptr<FidStorageBase> createStorage(int num, QString path="") override;
};


class FtmwConfigDuration : public FtmwConfig
{
public:
    FtmwConfigDuration(const QString& scopeHwKey);
    FtmwConfigDuration(const FtmwConfig &other);
    ~FtmwConfigDuration() {}

    // ExperimentObjective interface
    int perMilComplete() const override;
    bool isComplete() const override;

    // FtmwConfig interface
    quint64 completedShots() const override;
protected:
    bool _init() override;
    void _prepareToSave() override;
    void _loadComplete() override;
    std::shared_ptr<FidStorageBase> createStorage(int num, QString path="") override;

private:
    QDateTime d_startTime, d_targetTime;
};


class FtmwConfigForever : public FtmwConfig
{
public:
    FtmwConfigForever(const QString& scopeHwKey);
    FtmwConfigForever(const FtmwConfig &other);
    ~FtmwConfigForever() {}

    // ExperimentObjective interface
    int perMilComplete() const override;
    bool indefinite() const override;
    bool isComplete() const override;

    // FtmwConfig interface
    quint64 completedShots() const override;
protected:
    bool _init() override;
    void _prepareToSave() override;
    void _loadComplete() override;
    std::shared_ptr<FidStorageBase> createStorage(int num, QString path="") override;
};

namespace BC::Store::FtmwLO {
inline constexpr QLatin1StringView upStart{"UpLOBegin"};
inline constexpr QLatin1StringView upEnd{"UpLOEnd"};
inline constexpr QLatin1StringView upMin{"UpMinorSteps"};
inline constexpr QLatin1StringView upMaj{"UpMajorSteps"};
inline constexpr QLatin1StringView downStart{"DownLOBegin"};
inline constexpr QLatin1StringView downEnd{"DownLOEnd"};
inline constexpr QLatin1StringView downMin{"DownMinorSteps"};
inline constexpr QLatin1StringView downMaj{"DownMajorSteps"};
}

class FtmwConfigLOScan : public FtmwConfig
{
public:
    FtmwConfigLOScan(const QString& scopeHwKey);
    FtmwConfigLOScan(const FtmwConfig &other);
    ~FtmwConfigLOScan() {}

    double d_upStart{0.0}, d_upEnd{0.0}, d_downStart{0.0}, d_downEnd{0.0};
    int d_upMaj{0}, d_upMin{0}, d_downMaj{0}, d_downMin{0};

    // ExperimentObjective interface
    int perMilComplete() const override;
    bool isComplete() const override;

    // FtmwConfig interface
    quint64 completedShots() const override;
protected:
    bool _init() override;
    void _prepareToSave() override;
    void _loadComplete() override;
    std::shared_ptr<FidStorageBase> createStorage(int num, QString path) override;
};

namespace BC::Store::FtmwDR {
inline constexpr QLatin1StringView drStart{"DRBegin"};
inline constexpr QLatin1StringView drStep{"DRStep"};
inline constexpr QLatin1StringView drEnd{"DREnd"};
inline constexpr QLatin1StringView drNumSteps{"DRNumSteps"};
}

class FtmwConfigDRScan : public FtmwConfig
{
public:
    FtmwConfigDRScan(const QString& scopeHwKey);
    FtmwConfigDRScan(const FtmwConfig &other);
    ~FtmwConfigDRScan() {};

    double d_start{0.0}, d_step{1.0};
    int d_numSteps{2};

    // ExperimentObjective interface
    int perMilComplete() const override;
    bool isComplete() const override;

    // FtmwConfig interface
    quint64 completedShots() const override;

protected:
    bool _init() override;
    void _prepareToSave() override;
    void _loadComplete() override;
    std::shared_ptr<FidStorageBase> createStorage(int num, QString path) override;
};

#endif // FTMWCONFIGTYPES_H
