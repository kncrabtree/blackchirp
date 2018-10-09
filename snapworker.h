#ifndef SNAPWORKER_H
#define SNAPWORKER_H

#include <QObject>

#include "ftmwconfig.h"

class SnapWorker : public QObject
{
    Q_OBJECT
public:
    explicit SnapWorker(QObject *parent = 0);

signals:
    void fidsUpdated(const FtmwConfig);

public slots:
    void calculateSnapshots(FtmwConfig allFids, const QList<int> snaps, bool includeRemainder, int num, QString path);

};

#endif // SNAPWORKER_H



