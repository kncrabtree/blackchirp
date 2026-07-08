#ifndef LIFLASERSTATUSBOX_H
#define LIFLASERSTATUSBOX_H

#include <QString>

#include <gui/widget/hardwarestatusbox.h>
#include <data/lif/lifunits.h>

class QLabel;
class Led;

class LifLaserStatusBox : public HardwareStatusBox
{
    Q_OBJECT
public:
    LifLaserStatusBox(const QString &key, QWidget *parent = nullptr);

    void applySettings();
    //! \a d is the output-beam wavenumber (cm⁻¹), per LifLaser::laserPosUpdate.
    void setPosition(double d);
    void setFlashlampEnabled(bool en);

private:
    QLabel *p_posLabel;
    Led *p_led;
    int d_decimals{2};
    QString d_suffix;
    BC::LifConv::LaserUnit d_unit{BC::LifConv::LaserUnit::Nm};
    double d_position{0.0}; ///< Raw output-beam wavenumber (cm⁻¹); converted to d_unit for display.

};

#endif // LIFLASERSTATUSBOX_H
