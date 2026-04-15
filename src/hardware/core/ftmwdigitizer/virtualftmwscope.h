#ifndef VIRTUALFTMWSCOPE_H
#define VIRTUALFTMWSCOPE_H

#include <hardware/core/ftmwdigitizer/ftmwscope.h>

#include <QVector>
#include <QTimer>

namespace BC::Key::FtmwScope {
inline constexpr QLatin1StringView interval{"shotIntervalMs"};
}

class VirtualFtmwScope : public FtmwScope
{
    Q_OBJECT
public:
    explicit VirtualFtmwScope(const QString& label, QObject *parent = nullptr);
    ~VirtualFtmwScope();

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

#endif // VIRTUALFTMWSCOPE_H
