#ifndef VIRTUALIOBOARD_H
#define VIRTUALIOBOARD_H

#include <hardware/core/ioboard/ioboard.h>

namespace BC::Key::IOB {
static const QString viobName("Virtual IO Board");
}

class VirtualIOBoard : public IOBoard
{
    Q_OBJECT
public:
    explicit VirtualIOBoard(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;

    // HardwareObject interface
protected:
    bool testConnection() override;
    void initialize() override;
    virtual AuxDataStorage::AuxDataMap readAuxData() override;
    virtual AuxDataStorage::AuxDataMap readValidationData() override;

private:
    AuxDataStorage::AuxDataMap auxData(bool plot);

    // IOBoard interface
protected:
    void readIOBSettings() override;
};

#endif // VIRTUALIOBOARD_H
