#ifndef OSCILLOSCOPE_H
#define OSCILLOSCOPE_H

#include "tcpinstrument.h"
#include <QVector>
#include <QPointF>
#include <QDataStream>
#include <QTextStream>
#include <QPointF>
#include <QTime>
#include <QStringList>
#include "fid.h"
#include <QTimer>

class Oscilloscope : public TcpInstrument
{
    Q_OBJECT
public:
    explicit Oscilloscope(QObject *parent = 0);

    enum ResponseType {
        RawData,
        BlockData
    };

    static Fid parseWaveform(QByteArray b, const FtmwConfig::ScopeConfig &config, const double loFreq, const Fid::Sideband sb);

signals:
    void shotAcquired(const QByteArray data);

public slots:
    void initialize();
    bool testConnection();

    void readWaveform();
    void endAcquisition(bool unlock = true);

    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void queryScope();
    void wakeUp();

private:
    bool d_waitingForReply;
    bool d_foundHeader;
    int d_headerNumBytes;
    int d_waveformBytes;
    FtmwConfig::ScopeConfig d_configuration;
    QDateTime d_lastTrigger;
    bool d_waitingForWakeUp;
    QTimer d_scopeTimeout;

    QByteArray scopeQueryCmd(QString query);

    QByteArray makeSimulatedData();
    QVector<double> d_simulatedData;




};

#endif // OSCILLOSCOPE_H
