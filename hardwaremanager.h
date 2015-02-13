#ifndef HARDWAREMANAGER_H
#define HARDWAREMANAGER_H

#include <QObject>
#include "loghandler.h"

class HardwareManager : public QObject
{
    Q_OBJECT
public:
    explicit HardwareManager(QObject *parent = 0);
    ~HardwareManager();

signals:
    void logMessage(const QString, const LogHandler::MessageCode = LogHandler::Normal);
    void statusMessage(const QString);

public slots:
    void initialize();
};

#endif // HARDWAREMANAGER_H
