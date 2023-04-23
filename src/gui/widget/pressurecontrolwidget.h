#ifndef PRESSURECONTROLWIDGET_H
#define PRESSURECONTROLWIDGET_H

#include <QWidget>

#include <hardware/optional/pressurecontroller/pressurecontrollerconfig.h>

class QDoubleSpinBox;
class QPushButton;

class PressureControlWidget : public QWidget
{
    Q_OBJECT
public:
    explicit PressureControlWidget(QWidget *parent = nullptr);

    void initialize(const PressureControllerConfig &cfg);

signals:
    void setpointChanged(double);
    void pressureControlModeChanged(bool);
    void valveOpen();
    void valveClose();

public slots:
    void pressureSetpointUpdate(double p);
    void pressureControlModeUpdate(bool en);

private:
    QDoubleSpinBox *p_setpointBox;
    QPushButton *p_controlButton;

};

#endif // PRESSURECONTROLWIDGET_H
