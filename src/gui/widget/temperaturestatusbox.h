#ifndef TEMPERATURESTATUSBOX_H
#define TEMPERATURESTATUSBOX_H

#include "hardwarestatusbox.h"

class QLabel;
class QDoubleSpinBox;

class TemperatureStatusBox : public HardwareStatusBox
{
    Q_OBJECT
public:
    struct ChWidgets {
        QLabel *label;
        QDoubleSpinBox *box;
        bool active{false};
    };

    TemperatureStatusBox(QString key, QWidget *parent = nullptr);

public slots:
    void loadFromSettings();
    void setTemperature(int ch, double t);
    void setChannelName(int ch, const QString name);
    void setChannelEnabled(int ch, bool en);

private:
    std::vector<ChWidgets> d_widgets;
};

#endif // TEMPERATURESTATUSBOX_H
