#ifndef FTMWCONFIG_H
#define FTMWCONFIG_H

#include <QDateTime>
#include <QDataStream>
#include <QVariant>
#include <QMetaType>
#include <memory>

#include <data/storage/fidstoragebase.h>
#include <data/experiment/fid.h>
#include <data/experiment/rfconfig.h>
#include <data/experiment/ftmwdigitizerconfig.h>
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
}


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
    ~FtmwConfig();

    int d_duration{0};
    bool d_isEnabled{false};
    bool d_phaseCorrectionEnabled{false};
    bool d_chirpScoringEnabled{false};
    double d_chirpRMSThreshold{0.0};
    double d_chirpOffsetUs{-1.0};
    FtmwType d_type{Forever};
    quint64 d_targetShots{0};

    FtmwDigitizerConfig d_scopeConfig;
    RfConfig d_rfConfig;
    QString d_errorString;

    bool initialize() override;
    bool advance() override;
    void hwReady() override;
    int perMilComplete() const override;
    bool indefinite() const override;
    bool isComplete() const override;
    bool abort() override;


    quint64 completedShots() const;
    QDateTime targetTime() const;

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
    bool writeFids(int num, QString path = QString(""), int snapNum = -1) const;


#ifdef BC_CUDA
    bool setFidsData(const QVector<QVector<qint64> > newList);
#endif
    bool addFids(const QByteArray rawData, int shift = 0);
    bool subtractFids(const FtmwConfig other);
    void setScopeConfig(const FtmwDigitizerConfig &other);
    void finalizeSnapshots(int num, QString path = QString(""));
    std::shared_ptr<FidStorageBase> storage() const;

    void loadFids(const int num, const QString path = QString(""));
    void loadFidsFromSnapshots(const int num, const QString path = QString(""), const QList<int> snaps = QList<int>());
    void loadChirps(const int num, const QString path = QString(""));
    void loadClocks(const int num, const QString path = QString(""));

private:
    std::shared_ptr<FidStorageBase> p_fidStorage;
    QDateTime d_targetTime;
    Fid d_fidTemplate;
    bool d_processingPaused{false};

    // HeaderStorage interface
protected:
    void prepareToSave() override;
    void loadComplete() override;
};



#endif // FTMWCONFIG_H
