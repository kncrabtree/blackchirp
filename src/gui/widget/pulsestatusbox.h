#ifndef PULSESTATUSBOX_H
#define PULSESTATUSBOX_H

#include <QGroupBox>
#include <hardware/optional/pulsegenerator/pulsegenconfig.h>

class QLabel;
class Led;

class PulseStatusBox : public QGroupBox, public PulseGenConfig
{
    Q_OBJECT
public:
    explicit PulseStatusBox(QWidget *parent = nullptr);

public slots:
    void updatePulseLeds(const PulseGenConfig &cc);
    void updatePulseLed(int index,Setting s, QVariant val);
    void updateRepRate(double rr);
    void updatePGenMode(PulseGenConfig::PGenMode m);
    void updatePGenEnabled(bool en);

signals:

private:
    std::vector<std::pair<QLabel*,Led*>> d_ledList;
    QLabel *p_repLabel;
    Led *p_enLed;

    void updateAll();

};

#endif // PULSESTATUSBOX_H
