#ifndef VIRTUALFTMWSCOPE_H
#define VIRTUALFTMWSCOPE_H

#include <hardware/core/ftmwdigitizer/ftmwscope.h>

#include <QVector>
#include <QTimer>

namespace BC::Key::FtmwScope {
static const QString vftmwName("Virtual FTMW Oscilloscope");
static const QString interval{"shotIntervalMs"};
}

class VirtualFtmwScope : public FtmwScope
{
    Q_OBJECT
public:
    explicit VirtualFtmwScope(QObject *parent = nullptr);
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
};

#endif // VIRTUALFTMWSCOPE_H
