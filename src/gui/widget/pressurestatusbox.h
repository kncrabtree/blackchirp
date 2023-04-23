#ifndef PRESSURESTATUSBOX_H
#define PRESSURESTATUSBOX_H

#include <QGroupBox>

class QDoubleSpinBox;
class Led;

class PressureStatusBox : public QGroupBox
{
    Q_OBJECT
public:
    PressureStatusBox(QWidget *parent = nullptr);

public slots:
    void pressureUpdate(double p);
    void pressureControlUpdate(bool en);
    void updateFromSettings();

private:
    QDoubleSpinBox *p_cpBox;
    Led *p_led;
};

#endif // PRESSURESTATUSBOX_H
