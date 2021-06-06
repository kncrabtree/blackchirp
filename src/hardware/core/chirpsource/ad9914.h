#ifndef AD9914_H
#define AD9914_H

#include <src/hardware/core/chirpsource/awg.h>


class AD9914 : public AWG
{
    Q_OBJECT
public:
    explicit AD9914(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    void readSettings() override;
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;

private:
    QByteArray d_settingsHex;
    double d_clockFreqHz;

protected:
    void initialize() override;
    bool testConnection() override;
};

#endif // AD9914_H
