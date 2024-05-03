#ifndef PULSEGENERATORMANAGER_H
#define PULSEGENERATORMANAGER_H

#include <QObject>

#include <data/storage/settingsstorage.h>
#include <data/experiment/experiment.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>

/**
 * @brief The PulseGeneratorManager class keeps track of all pulse generators and any coordination among them.
 *
 * In addition to ensuring that Blackchirp can track which pulse generators
 * might trigger others in order to get a representative timing diagram, some
 * channels can be identified by their logical role (e.g., discharge pulse).
 * This class lets other parts of the program make settings to those channels
 * without requiring knowledge of which physical pulse generator they're assigned to.
 */
class PulseGeneratorManager : public QObject, public SettingsStorage
{
    Q_OBJECT
public:
    explicit PulseGeneratorManager(QObject *parent = nullptr);

signals:
    void logMessage(QString, LogHandler::MessageCode mc = LogHandler::Normal);

private:
    QVector<PulseGenerator*> d_pGenList;

    friend class HardwareManager;

};

#endif // PULSEGENERATORMANAGER_H
