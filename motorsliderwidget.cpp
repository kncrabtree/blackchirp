#include "motorsliderwidget.h"

#include <QVBoxLayout>

MotorSliderWidget::MotorSliderWidget(QWidget *parent) : QWidget(parent)
{

    QVBoxLayout *l = new QVBoxLayout();

    p_label = new QLabel();
    p_label->setAlignment(Qt::AlignCenter);
    l->addWidget(p_label,0,Qt::AlignCenter);

    p_slider = new QSlider(Qt::Vertical);
    p_slider->setRange(0,1);
    p_slider->setSizePolicy(QSizePolicy::Preferred,QSizePolicy::Expanding);
    l->addWidget(p_slider,1,Qt::AlignHCenter);

    p_dsb = new QDoubleSpinBox();
    p_dsb->setRange(0.0,1.0);
    p_dsb->setSuffix(QString(""));
    p_dsb->setKeyboardTracking(false);
    l->addWidget(p_dsb,0);

    setLayout(l);

    setAxis(BlackChirp::MotorX);

    connect(p_slider,&QSlider::valueChanged,this,&MotorSliderWidget::valueChanged);
    connect(p_slider,&QSlider::valueChanged,this,&MotorSliderWidget::updateBox);
    connect(p_dsb,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),this,&MotorSliderWidget::updateSlider);

}

BlackChirp::MotorAxis MotorSliderWidget::axis() const
{
    return d_currentAxis;
}

int MotorSliderWidget::currentIndex() const
{
    return p_slider->value();
}

void MotorSliderWidget::changeAxis(BlackChirp::MotorAxis a, const MotorScan s)
{
    setAxis(a);
    setRange(s);
}

void MotorSliderWidget::setRange(const MotorScan s)
{
    QPair<double,double> range = s.range(d_currentAxis);
    int steps = s.numPoints(d_currentAxis);

    double theRange = fabs(range.first-range.second);
    int decimals = 1;
    if(theRange < 10.0)
        decimals = 2;
    if(theRange < 1.0)
        decimals = 3;

    setRange(range.first,range.second,steps,decimals);
    p_slider->setValue(steps/2);
}

void MotorSliderWidget::setRange(double min, double max, int steps, int decimals)
{
    if(max < min)
        qSwap(max,min);

    d_min = min;
    d_max = max;
    d_stepSize = (max - min)/((double)steps-1.0);
    d_numSteps = steps;

    p_slider->setRange(0,steps-1);

    p_dsb->setDecimals(decimals);
    p_dsb->setRange(min,max);
    p_dsb->setSingleStep(d_stepSize);

    p_slider->setValue(steps/2);
}

void MotorSliderWidget::updateSlider(double newVal)
{
    //need to find closest step to newVal, the update double spin box accordingly
    int step = qBound(0,qRound((newVal - d_min)/d_stepSize),d_numSteps-1);

    p_slider->blockSignals(true);
    p_slider->setValue(step);
    p_slider->blockSignals(false);

    updateBox(step);
    emit valueChanged(step);
}

void MotorSliderWidget::updateBox(int newVal)
{
    p_dsb->blockSignals(true);
    p_dsb->setValue(d_min + (double)d_stepSize*newVal);
    p_dsb->blockSignals(false);
}

void MotorSliderWidget::setLabel(QString s)
{
    p_label->setText(s);
}

void MotorSliderWidget::setUnits(QString u)
{
    p_dsb->setSuffix(QString(" ")+u);
}

void MotorSliderWidget::setAxis(BlackChirp::MotorAxis a)
{
    switch(a)
    {
    case BlackChirp::MotorX:
        setLabel(QString("X"));
        setUnits(QString("mm"));
        break;
    case BlackChirp::MotorY:
        setLabel(QString("Y"));
        setUnits(QString("mm"));
        break;
    case BlackChirp::MotorZ:
        setLabel(QString("Z"));
        setUnits(QString("mm"));
        break;
    case BlackChirp::MotorT:
        setLabel(QString("T"));
        setUnits(QString::fromUtf16(u"µs"));
        break;
    }

    d_currentAxis = a;
}

