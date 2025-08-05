#ifndef IOBOARD_H
#define IOBOARD_H

#include <hardware/core/hardwareobject.h>

#include <data/experiment/hardware/optional/ioboard/ioboardconfig.h>


class IOBoard : public HardwareObject, public IOBoardConfig
{
    Q_OBJECT
public:
    explicit IOBoard(const QString& impl, const QString& label, QObject *parent = nullptr);
    virtual ~IOBoard();

    virtual QStringList validationKeys() const override;

protected:
    virtual std::map<int,double> readAnalogChannels() =0;
    virtual std::map<int,bool> readDigitalChannels() =0;

private:
    AuxDataStorage::AuxDataMap readAuxData() override;
    AuxDataStorage::AuxDataMap readValidationData() override;
    void writeSettings();


    // HardwareObject interface
public slots:
    bool hwPrepareForExperiment(Experiment &exp) override final;
    IOBoardConfig getConfig() { return static_cast<IOBoardConfig&>(*this); }

    // HardwareObject interface
public slots:
    QStringList forbiddenKeys() const override;
};

#endif // IOBOARD_H
