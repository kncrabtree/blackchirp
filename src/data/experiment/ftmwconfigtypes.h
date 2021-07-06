#ifndef FTMWCONFIGTYPES_H
#define FTMWCONFIGTYPES_H

#include <data/experiment/ftmwconfig.h>
#include <QDateTime>

class FtmwConfigSingle : public FtmwConfig
{
public:
    FtmwConfigSingle() {}
    FtmwConfigSingle(const FtmwConfig &other);
    ~FtmwConfigSingle() {}

    // ExperimentObjective interface
    int perMilComplete() const override;
    bool isComplete() const override;

    // FtmwConfig interface
protected:
    bool _init() override;
    void _prepareToSave() override;
    void _loadComplete() override;
    std::shared_ptr<FidStorageBase> createStorage() override;

private:
    quint64 d_targetShots;
};


class FtmwConfigPeakUp : public FtmwConfig
{
public:
    FtmwConfigPeakUp() {}
    FtmwConfigPeakUp(const FtmwConfig &other);
    ~FtmwConfigPeakUp() {}

    // ExperimentObjective interface
    int perMilComplete() const override;
    bool isComplete() const override;

    // FtmwConfig interface
protected:
    quint8 bitShift() const override;
    bool _init() override;
    void _prepareToSave() override;
    void _loadComplete() override;
    std::shared_ptr<FidStorageBase> createStorage() override;

private:
    quint64 d_targetShots;
};


class FtmwConfigDuration : public FtmwConfig
{
public:
    FtmwConfigDuration() {}
    FtmwConfigDuration(const FtmwConfig &other);
    ~FtmwConfigDuration() {}

    // ExperimentObjective interface
    int perMilComplete() const override;
    bool isComplete() const override;

    // FtmwConfig interface
protected:
    bool _init() override;
    void _prepareToSave() override;
    void _loadComplete() override;
    std::shared_ptr<FidStorageBase> createStorage() override;

private:
    QDateTime d_startTime, d_targetTime;
};


class FtmwConfigForever : public FtmwConfig
{
public:
    FtmwConfigForever() {}
    FtmwConfigForever(const FtmwConfig &other);
    ~FtmwConfigForever() {}

    // ExperimentObjective interface
    int perMilComplete() const override;
    bool indefinite() const override;
    bool isComplete() const override;

    // FtmwConfig interface
protected:
    bool _init() override;
    void _prepareToSave() override;
    void _loadComplete() override;
    std::shared_ptr<FidStorageBase> createStorage() override;
};

#endif // FTMWCONFIGTYPES_H
