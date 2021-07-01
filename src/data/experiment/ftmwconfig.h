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


class FtmwConfig : public ExperimentObjective
{
public:
    FtmwConfig() {};
    FtmwConfig(const FtmwConfig &) =default;
    FtmwConfig &operator=(const FtmwConfig &) =default;
    ~FtmwConfig();

    bool initialize() override;
    bool advance() override;
    void hwReady() override;
    int perMilComplete() const override;
    bool indefinite() const override;
    bool isComplete() const override;
    bool abort() override;

    bool isEnabled() const;
    bool isPhaseCorrectionEnabled() const;
    bool isChirpScoringEnabled() const;
    double chirpRMSThreshold() const;
    double chirpOffsetUs() const;
    BlackChirp::FtmwType type() const;
    quint64 targetShots() const;
    quint64 completedShots() const;
    QDateTime targetTime() const;
//    QVector<qint64> rawFidList() const;
//    QList<FidList> multiFidList() const;
    FtmwDigitizerConfig scopeConfig() const;
    RfConfig rfConfig() const;
    ChirpConfig chirpConfig(int num = 0) const;
    Fid fidTemplate() const;
    bool processingPaused() const;
    int numFrames() const;
    int numSegments() const;
    quint64 shotIncrement() const;
    FidList parseWaveform(const QByteArray b) const;
    QVector<qint64> extractChirp() const;
    QVector<qint64> extractChirp(const QByteArray b) const;
    QString errorString() const;
    double ftMinMHz() const;
    double ftMaxMHz() const;
    double ftNyquistMHz() const;
    double fidDurationUs() const;
    QPair<int,int> chirpRange() const;
    bool writeFids(int num, QString path = QString(""), int snapNum = -1) const;

    void setEnabled(bool en = true);
    void setPhaseCorrectionEnabled(bool enabled);
    void setChirpScoringEnabled(bool enabled);
    void setChirpRMSThreshold(double t);
    void setChirpOffsetUs(double o);
    void setFidTemplate(const Fid f);
    void setType(const BlackChirp::FtmwType type);
    void setTargetShots(const qint64 target);
    void setTargetTime(const QDateTime time);
#ifdef BC_CUDA
    bool setFidsData(const QVector<QVector<qint64> > newList);
#endif
    bool addFids(const QByteArray rawData, int shift = 0);
    bool subtractFids(const FtmwConfig other);
    void resetFids();
    void setScopeConfig(const FtmwDigitizerConfig &other);
    void setRfConfig(const RfConfig other);
    void finalizeSnapshots(int num, QString path = QString(""));
    std::shared_ptr<FidStorageBase> storage();


    QMap<QString,QPair<QVariant,QString> > headerMap() const;
    void loadFids(const int num, const QString path = QString(""));
    void loadFidsFromSnapshots(const int num, const QString path = QString(""), const QList<int> snaps = QList<int>());
    void parseLine(const QString key, const QVariant val);
    void loadChirps(const int num, const QString path = QString(""));
    void loadClocks(const int num, const QString path = QString(""));


    void saveToSettings() const;
    static FtmwConfig loadFromSettings();

private:
    bool d_isEnabled{false};
    bool d_phaseCorrectionEnabled{false};
    bool d_chirpScoringEnabled{false};
    double d_chirpRMSThreshold{0.0};
    double d_chirpOffsetUs{-1.0};
    BlackChirp::FtmwType d_type{BlackChirp::FtmwForever};
    quint64 d_targetShots{0};
//    quint64 d_completedShots{0};
    QDateTime d_targetTime;
    std::shared_ptr<FidStorageBase> p_fidStorage;

//    bool d_multipleFidLists{false};
//    FidList d_fidList;
//    QList<FidList> d_multiFidStorage;

    FtmwDigitizerConfig d_scopeConfig;
    RfConfig d_rfConfig;
    Fid d_fidTemplate;
    QString d_errorString;

    bool d_processingPaused{false};
};



#endif // FTMWCONFIG_H
