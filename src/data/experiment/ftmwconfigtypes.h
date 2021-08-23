#ifndef FTMWCONFIGTYPES_H
#define FTMWCONFIGTYPES_H

#include <data/experiment/ftmwconfig.h>
#include <QDateTime>

class FtmwConfigSingle : public FtmwConfig
{
public:
    FtmwConfigSingle();
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
    FtmwConfigPeakUp();
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
    FtmwConfigDuration();
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
    FtmwConfigForever();
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
static const QString upStart("UpLOBegin");
static const QString upEnd("UpLOEnd");
static const QString upMin("UpMinorSteps");
static const QString upMaj("UpMajorSteps");
static const QString downStart("DownLOBegin");
static const QString downEnd("DownLOEnd");
static const QString downMin("DownMinorSteps");
static const QString downMaj("DownMajorSteps");
}

class FtmwConfigLOScan : public FtmwConfig
{
public:
    FtmwConfigLOScan();
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
static const QString drStart{"DRBegin"};
static const QString drStep{"DRStep"};
static const QString drEnd{"DREnd"};
static const QString drNumSteps{"DRNumSteps"};
}

class FtmwConfigDRScan : public FtmwConfig
{
public:
    FtmwConfigDRScan();
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
