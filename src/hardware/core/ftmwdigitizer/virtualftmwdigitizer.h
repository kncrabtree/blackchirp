#ifndef VIRTUALFTMWDIGITIZER_H
#define VIRTUALFTMWDIGITIZER_H

#include <hardware/core/ftmwdigitizer/ftmwdigitizer.h>

#include <QVector>
#include <QTimer>

namespace BC::Key::FtmwDigitizer {
inline constexpr QLatin1StringView interval{"shotIntervalMs"};
}

class VirtualFtmwDigitizer : public FtmwDigitizer
{
    Q_OBJECT
public:
    explicit VirtualFtmwDigitizer(const QString& label, QObject *parent = nullptr);
    ~VirtualFtmwDigitizer();

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;

    void readWaveform() override;

protected:
    bool testConnection() override;
    void initialize() override;

private:
    QVector<double> d_simulatedData;
    QTimer *d_simulatedTimer = nullptr;
    QTime d_testTime;
    
    void generateSimulatedFid();
};

#endif // VIRTUALFTMWDIGITIZER_H
