#ifndef AD9914_H
#define AD9914_H

#include <hardware/optional/chirpsource/awg.h>

namespace BC::Key::AWG {
static const QString ad9914{"ad9914"};
static const QString ad9914Name("AD9914 Direct Digital Synthesizer");
}

class AD9914 : public AWG
{
    Q_OBJECT
public:
    explicit AD9914(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;

private:
    QByteArray d_settingsHex;

protected:
    void initialize() override;
    bool testConnection() override;
};

#endif // AD9914_H
