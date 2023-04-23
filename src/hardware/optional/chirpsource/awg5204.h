#ifndef AWG5204_H
#define AWG5204_H

#include <hardware/optional/chirpsource/awg.h>

namespace BC::Key::AWG {
static const QString awg5204{"awg5204"};
static const QString awg5204Name("Arbitrary Waveform Generator AWG5204");
}


class AWG5204 : public AWG
{
    Q_OBJECT
public:
    explicit AWG5204(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;

protected:
    void initialize() override;
    bool testConnection() override;

private:
    QString getWaveformKey(const ChirpConfig cc);
    QString writeWaveform(const ChirpConfig cc);
};

#endif // AWG5204_H
