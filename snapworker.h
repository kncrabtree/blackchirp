#ifndef SNAPWORKER_H
#define SNAPWORKER_H

#include <QObject>

#include "fid.h"

class SnapWorker : public QObject
{
    Q_OBJECT
public:
    explicit SnapWorker(QObject *parent = 0);
    FidList parseFile(int exptNum, int snapNum = -1, QString path = QString(""));

signals:
    void fidListComplete(const FidList);

public slots:
    void calculateFidList(int exptNum, const FidList allList, const QList<int> snapList, bool subtractFromFull);

};

#endif // SNAPWORKER_H
