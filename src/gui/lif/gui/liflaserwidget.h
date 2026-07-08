#ifndef LIFLASERWIDGET_H
#define LIFLASERWIDGET_H

#include <QWidget>

#include <data/lif/lifunits.h>

class QDoubleSpinBox;
class QPushButton;

class LifLaserWidget : public QWidget
{
    Q_OBJECT
public:
    explicit LifLaserWidget(const QString& lifLaserKey, QWidget *parent = nullptr);

    //! \a d is the output-beam wavenumber (cm⁻¹), per LifLaser::laserPosUpdate.
    void setPosition(const double d);
    void setFlashlamp(bool b);

signals:
    //! Output-beam wavenumber (cm⁻¹); dispatched to LifLaser::setPosition.
    void changePosition(double);
    void changeFlashlamp(bool);

private:
    QDoubleSpinBox *p_posBox;
    QPushButton *p_posSetButton;
    QPushButton *p_flButton;
    BC::LifConv::LaserUnit d_unit{BC::LifConv::LaserUnit::Nm};
};

#endif // LIFLASERWIDGET_H
