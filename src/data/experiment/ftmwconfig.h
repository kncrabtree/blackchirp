#ifndef FTMWCONFIG_H
#define FTMWCONFIG_H

#include <QDateTime>
#include <QVariant>
#include <QMetaType>
#include <memory>

#include <data/storage/fidstoragebase.h>
#include <data/experiment/fid.h>
#include <data/experiment/rfconfig.h>
#include <hardware/core/ftmwdigitizer/ftmwdigitizerconfig.h>
#include <data/datastructs.h>
#include <data/experiment/experimentobjective.h>

#define BC_FTMW_MAXSHIFT 50

namespace BC::Store::FTMW {
static const QString key("FtmwConfig");
static const QString duration("Duration");
static const QString enabled("Enabled");
static const QString phase("PhaseCorrectionEnabled");
static const QString chirp("ChirpScoringEnabled");
static const QString chirpThresh("ChirpRMSThreshold");
static const QString chirpOffset("ChirpOffset");
static const QString type("Type");
static const QString tShots("TargetShots");
static const QString objective("Objective");
}

class BlackchirpCSV;

class FtmwConfig : public ExperimentObjective, public HeaderStorage
{
    Q_GADGET
public:
    enum FtmwType
    {
        Target_Shots,
        Target_Duration,
        Forever,
        Peak_Up,
        LO_Scan,
        DR_Scan
    };
    Q_ENUM(FtmwType)

    FtmwConfig();
    FtmwConfig(const FtmwConfig &) =default;
    FtmwConfig &operator=(const FtmwConfig &) =default;
    virtual ~FtmwConfig();

    bool d_phaseCorrectionEnabled{false};
    bool d_chirpScoringEnabled{false};
    double d_chirpRMSThreshold{0.0};
    double d_chirpOffsetUs{-1.0};
    FtmwType d_type{Forever};
    quint64 d_objective{0};

    FtmwDigitizerConfig d_scopeConfig;
    RfConfig d_rfConfig;
    QString d_errorString;

    bool initialize() override;
    bool advance() override;
    void hwReady() override;
    bool abort() override;

    virtual bool indefinite() const override { return false; }

    quint64 completedShots() const;

    bool processingPaused() const;
    quint64 shotIncrement() const;
    FidList parseWaveform(const QByteArray b) const;
    QVector<qint64> extractChirp() const;
    QVector<qint64> extractChirp(const QByteArray b) const;
    double ftMinMHz() const;
    double ftMaxMHz() const;
    double ftNyquistMHz() const;
    double fidDurationUs() const;
    QPair<int,int> chirpRange() const;

#ifdef BC_CUDA
    bool setFidsData(const QVector<QVector<qint64> > newList);
#endif
    bool addFids(const QByteArray rawData, int shift = 0);
    void setScopeConfig(const FtmwDigitizerConfig &other);
    std::shared_ptr<FidStorageBase> storage() const;

    void loadFids(int num, QString path = QString(""));

private:
    std::shared_ptr<FidStorageBase> p_fidStorage;
    Fid d_fidTemplate;
    bool d_processingPaused{false};
    QDateTime d_lastAutosaveTime;

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
    void prepareChildren() override;

    virtual quint8 bitShift() const { return 0; }
    virtual bool _init() =0;
    virtual void _prepareToSave() =0;
    virtual void _loadComplete() =0;
    virtual std::shared_ptr<FidStorageBase> createStorage(int num, QString path="") =0;
};



#endif // FTMWCONFIG_H
