#ifndef M8190_H
#define M8190_H

#include <hardware/optional/chirpsource/awg.h>

namespace BC {
namespace Key {
static const QString m8190{"m8190"};
static const QString m8190Name("Arbitrary Waveform Generator M8190");
}
}

/*!
 * \brief The M8190 class
 *
 * The chirp is sent to output 1; protection to marker 1, amp gate to marker 2
 *
 */
class M8190 : public AWG
{
    Q_OBJECT
public:
    explicit M8190(QObject *parent=nullptr);

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;

protected:
    bool testConnection() override;
    void initialize() override;

private:
    bool m8190Write(const QString cmd);
};

#endif // M8190_H
