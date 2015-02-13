#ifndef HARDWAREMANAGER_H
#define HARDWAREMANAGER_H

#include <QObject>

class HardwareManager : public QObject
{
    Q_OBJECT
public:
    explicit HardwareManager(QObject *parent = 0);
    ~HardwareManager();

signals:

public slots:
};

#endif // HARDWAREMANAGER_H
