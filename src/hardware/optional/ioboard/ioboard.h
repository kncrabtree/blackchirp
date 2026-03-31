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
    /*!
     * \brief Apply and validate channel/digitizer configuration on hardware.
     *
     * Called by hwPrepareForExperiment() with the desired configuration
     * (from the Experiment if present, or the current internal state).
     * Implementations should:
     *   1. Apply the settings in \p config to the hardware
     *   2. Read back actual values and update \p config with validated settings
     *   3. Return true on success, false on failure
     *
     * On success, the base class copies the (potentially modified) config
     * back to internal state and stores it in the Experiment.
     *
     * \param config The desired configuration (mutable — update with actual values)
     * \return true if configuration was applied successfully
     */
    virtual bool configure(IOBoardConfig &config) =0;

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
