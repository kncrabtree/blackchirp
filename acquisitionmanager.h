#ifndef ACQUISITIONMANAGER_H
#define ACQUISITIONMANAGER_H

#include <QObject>
#include "loghandler.h"

class AcquisitionManager : public QObject
{
    Q_OBJECT
public:
    explicit AcquisitionManager(QObject *parent = 0);
    ~AcquisitionManager();

signals:
    void logMessage(const QString,const LogHandler::MessageCode = LogHandler::Normal);
    void statusMessage(const QString);

public slots:
};

#endif // ACQUISITIONMANAGER_H
