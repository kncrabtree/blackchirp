#ifndef SNAPWORKER_H
#define SNAPWORKER_H

#include <QObject>

#include "fid.h"

class SnapWorker : public QObject
{
    Q_OBJECT
public:
    explicit SnapWorker(QObject *parent = 0);
    QList<Fid> parseFile(int exptNum, int snapNum = -1);

signals:
    void fidListComplete(const QList<Fid>);

public slots:
    void calculateFidList(int exptNum, const QList<int> snapList, bool subtractFromFull);

};

#endif // SNAPWORKER_H
