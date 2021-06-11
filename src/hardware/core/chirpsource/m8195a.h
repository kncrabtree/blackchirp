#ifndef M8195A_H
#define M8195A_H

#include <src/hardware/core/chirpsource/awg.h>

namespace BC {
namespace Key {
static const QString m8195a("m8195a");
static const QString m8195aName("Arbitrary Waveform Generator M8195A");
}
}

class M8195A : public AWG
{
    Q_OBJECT
public:
    explicit M8195A(QObject *parent=nullptr);

    // HardwareObject interface
public slots:
    void readSettings() override;
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
