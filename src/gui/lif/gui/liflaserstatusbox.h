#ifndef LIFLASERSTATUSBOX_H
#define LIFLASERSTATUSBOX_H

#include <QGroupBox>

#include <gui/widget/hardwarestatusbox.h>

class QDoubleSpinBox;
class Led;

class LifLaserStatusBox : public HardwareStatusBox
{
    Q_OBJECT
public:
    LifLaserStatusBox(QWidget *parent = nullptr);

    void applySettings();
    void setPosition(double d);
    void setFlashlampEnabled(bool en);

private:
    QDoubleSpinBox *p_posBox;
    Led *p_led;

};

#endif // LIFLASERSTATUSBOX_H
