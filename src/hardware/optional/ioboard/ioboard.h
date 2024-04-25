#ifndef IOBOARD_H
#define IOBOARD_H

#include <hardware/core/hardwareobject.h>

#include <hardware/optional/ioboard/ioboardconfig.h>

namespace BC::Key::IOB {
static const QString ioboard{"IOBoard"};
}

class IOBoard : public HardwareObject, public IOBoardConfig
{
    Q_OBJECT
public:
    explicit IOBoard(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent = nullptr, bool threaded=true, bool critical=false);
    virtual ~IOBoard();

    virtual QStringList validationKeys() const override;

protected:
    virtual std::map<int,double> readAnalogChannels() =0;
    virtual std::map<int,bool> readDigitalChannels() =0;

private:
    AuxDataStorage::AuxDataMap readAuxData() override;
    AuxDataStorage::AuxDataMap readValidationData() override;

    inline static int d_count = 0;

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;
    IOBoardConfig getConfig() { return static_cast<IOBoardConfig&>(*this); }

    // HardwareObject interface
public slots:
    QStringList forbiddenKeys() const override;
};

#endif // IOBOARD_H
