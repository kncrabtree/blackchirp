#ifndef AWG7122B_H
#define AWG7122B_H

#include <hardware/core/chirpsource/awg.h>

#include <data/experiment/chirpconfig.h>

namespace BC::Key::AWG {
static const QString awg7122b("awg7122b");
static const QString awg7122bName("Arbirtary Waveform Generator AWG7122B");
}

/*!
 * \brief The AWG7122B class
 *
 * Chirp is sent to output 1; Protection to marker 1, amp gate to marker 2
 */
class AWG7122B : public AWG
{
    Q_OBJECT
public:
    explicit AWG7122B(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;

protected:
    bool testConnection() override;
    void initialize() override;


private:
    QString getWaveformKey(const ChirpConfig cc);
    QString writeWaveform(const ChirpConfig cc);

    bool d_triggered;
};

#endif // AWG7122B_H
