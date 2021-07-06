#ifndef SNAPWORKER_H
#define SNAPWORKER_H

#include <QObject>

#include <data/experiment/ftmwconfig.h>

class SnapWorker : public QObject
{
    Q_OBJECT
public:
    explicit SnapWorker(QObject *parent = 0);

signals:
//    void processingComplete(const FtmwConfig);
//    void finalProcessingComplete(const FtmwConfig);

public slots:
//    FtmwConfig calculateSnapshots(FtmwConfig allFids, const QList<int> snaps, bool includeRemainder, int num, QString path);
//    void finalizeSnapshots(FtmwConfig allFids, const QList<int> snaps, bool includeRemainder, int num, QString path);

};

#endif // SNAPWORKER_H



