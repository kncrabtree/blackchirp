#ifndef MOTORSLIDERWIDGET_H
#define MOTORSLIDERWIDGET_H

#include <QWidget>

#include <QLabel>
#include <QSlider>
#include <QDoubleSpinBox>

class MotorSliderWidget : public QWidget
{
    Q_OBJECT
public:
    explicit MotorSliderWidget(QWidget *parent = 0);

signals:
    void valueChanged(int);

public slots:
    void setRange(double min, double max, int steps, int decimals = 2);
    void updateSlider(double newVal);
    void updateBox(int newVal);
    void setLabel(QString s);
    void setUnits(QString u);

protected:
    QLabel *p_label;
    QSlider *p_slider;
    QDoubleSpinBox *p_dsb;

    double d_min, d_max, d_stepSize;
    int d_numSteps;
};

#endif // MOTORSLIDERWIDGET_H
