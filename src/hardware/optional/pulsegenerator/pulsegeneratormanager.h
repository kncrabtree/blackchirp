#ifndef PULSEGENERATORMANAGER_H
#define PULSEGENERATORMANAGER_H

#include <QObject>

#include <data/storage/settingsstorage.h>
#include <data/experiment/experiment.h>

class PulseGeneratorManager : public QObject, public SettingsStorage
{
    Q_OBJECT
public:
    explicit PulseGeneratorManager(QObject *parent = nullptr);

signals:
    void logMessage(QString, LogHandler::MessageCode mc = LogHandler::Normal);
};

#endif // PULSEGENERATORMANAGER_H
