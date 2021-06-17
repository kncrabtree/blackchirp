#ifndef MOTORSLIDERWIDGET_H
#define MOTORSLIDERWIDGET_H

#include <QWidget>

#include <QLabel>
#include <QSlider>
#include <QDoubleSpinBox>

#include <src/modules/motor/data/motorscan.h>

class MotorSliderWidget : public QWidget
{
    Q_OBJECT
public:
    explicit MotorSliderWidget(QWidget *parent = 0);
    MotorScan::MotorAxis axis() const;
    int currentIndex() const;

signals:
    void valueChanged(int);

public slots:
    void changeAxis(MotorScan::MotorAxis a, const MotorScan s);
    void setRange(const MotorScan s);
    void setAxis(MotorScan::MotorAxis a);
    void updateSlider(double newVal);
    void updateBox(int newVal);

protected:
    QLabel *p_label;
    QSlider *p_slider;
    QDoubleSpinBox *p_dsb;
    MotorScan::MotorAxis d_currentAxis;

    double d_min, d_max, d_stepSize;
    int d_numSteps;

    void setLabel(QString s);
    void setUnits(QString u);
    void setRange(double min, double max, int steps, int decimals = 2);
};

#endif // MOTORSLIDERWIDGET_H
