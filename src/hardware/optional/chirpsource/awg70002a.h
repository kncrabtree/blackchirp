#ifndef AWG70002A_H
#define AWG70002A_H

#include <hardware/optional/chirpsource/awg.h>

namespace BC::Key::AWG {
static const QString awg70002a("awg70002a");
static const QString awg70002aName("Arbitrary Waveform Generator AWG70002A");
}

/*!
 * \brief The AWG70002a class
 *
 * Chirp is sent to output 1; Protection to marker 1, amp gate to marker 2
 *
 */
class AWG70002a : public AWG
{
    Q_OBJECT
public:
    explicit AWG70002a(QObject *parent = nullptr);

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
};

#endif // AWG70002A_H
