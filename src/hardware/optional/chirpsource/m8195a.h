#ifndef M8195A_H
#define M8195A_H

#include <hardware/optional/chirpsource/awg.h>

namespace BC {
namespace Key {
static const QString m8195a("m8195a");
static const QString m8195aName("Arbitrary Waveform Generator M8195A");
}
}

/*!
 * \brief The M8195A class
 *
 * The chirp is sent to output 1; protection to output 3, amp gate to output 4
 *
 */
class M8195A : public AWG
{
    Q_OBJECT
public:
    explicit M8195A(QObject *parent=nullptr);

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;

protected:
    bool testConnection() override;
    void initialize() override;


private:
    bool m8195aWrite(const QString cmd);
};

#endif // M8195A_H
