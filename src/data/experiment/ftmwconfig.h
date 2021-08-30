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
#include <data/experiment/experimentobjective.h>

#ifdef BC_CUDA
#include <modules/cuda/gpuaverager.h>
#endif

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
    virtual void cleanup() override;

    virtual bool indefinite() const override { return false; }

    virtual quint64 completedShots() const =0;

    bool processingPaused() const;
    quint64 shotIncrement() const;
    FidList parseWaveform(const QByteArray b) const;
    double ftMinMHz() const;
    double ftMaxMHz() const;
    double ftNyquistMHz() const;
    double fidDurationUs() const;
    double chirpFOM() const { return d_lastFom; };
    double chirpShift() const { return d_currentShift; }
    double chirpRMS() const { return d_lastRMS; }

    /*!
     * \brief Calculate first sample and num samples of chirp in FID record
     *
     * This function uses the duration of the first chirp and the scope sample
     * rate to compute the number of samples in the chirp. The start of the chirp
     * is either specified manually (d_chirpOffsetUs) or adding up the pre-chirp
     * protection and gate delays and subtracting the scope trigger offset.
     *
     * \return QPair<int,int> : The first integer is the index of the chirp start,
     * and the second is the number of samples
     */
    QPair<int,int> chirpRange() const;


    bool setFidsData(const QVector<QVector<qint64> > newList);
    bool addFids(const QByteArray rawData);
    void setScopeConfig(const FtmwDigitizerConfig &other);
    std::shared_ptr<FidStorageBase> storage() const;

    void loadFids(int num, QString path = QString(""));

private:
    std::shared_ptr<FidStorageBase> p_fidStorage;
    Fid d_fidTemplate;
    bool d_processingPaused{false};
    QDateTime d_lastAutosaveTime;
    int d_currentShift{0};
    float d_lastFom{0.0};
    double d_lastRMS{0.0};

    bool preprocessChirp(const FidList l);
    float calculateFom(const QVector<qint64> vec, const Fid fid, QPair<int,int> range, int trialShift);
    double calculateChirpRMS(const QVector<qint64> chirp, quint64 shots = 1);

#ifdef BC_CUDA
    std::shared_ptr<GpuAverager> ps_gpu;
#endif

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
