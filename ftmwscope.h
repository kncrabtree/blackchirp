#ifndef FTMWSCOPE_H
#define FTMWSCOPE_H

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

class FtmwScope : public TcpInstrument
{
    Q_OBJECT
public:
    explicit FtmwScope(QObject *parent = 0);

    enum ResponseType {
        RawData,
        BlockData
    };

signals:
    void shotAcquired(const QByteArray data);

public slots:
    void initialize();
    bool testConnection();

    void readWaveform();

    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
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
    QTimer d_simulatedTimer;
    QTime d_testTime;




};

#endif // FTMWSCOPE_H
