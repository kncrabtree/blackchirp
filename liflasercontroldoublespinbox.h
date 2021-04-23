#ifndef LIFLASERCONTROLDOUBLESPINBOX_H
#define LIFLASERCONTROLDOUBLESPINBOX_H

#include <QDoubleSpinBox>

class LifLaserControlDoubleSpinBox : public QDoubleSpinBox
{
public:
    LifLaserControlDoubleSpinBox(QWidget *parent = nullptr);

public slots:
    void configure(bool step=false);
};

#endif // LIFLASERCONTROLDOUBLESPINBOX_H
