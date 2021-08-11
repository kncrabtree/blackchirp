#ifndef PULSESTATUSBOX_H
#define PULSESTATUSBOX_H

#include <QGroupBox>
#include <hardware/optional/pulsegenerator/pulsegenconfig.h>

class QLabel;
class Led;

class PulseStatusBox : public QGroupBox
{
    Q_OBJECT
public:
    explicit PulseStatusBox(QWidget *parent = nullptr);

public slots:
    void updatePulseLeds(const PulseGenConfig &cc);
    void updatePulseLed(int index,PulseGenConfig::Setting s, QVariant val);
    void updateFromSettings();

signals:

private:
    std::vector<std::pair<QLabel*,Led*>> d_ledList;

};

#endif // PULSESTATUSBOX_H
