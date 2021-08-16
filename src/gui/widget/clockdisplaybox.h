#ifndef CLOCKDISPLAYBOX_H
#define CLOCKDISPLAYBOX_H

#include <QGroupBox>

#include <data/experiment/rfconfig.h>

class QDoubleSpinBox;

class ClockDisplayBox : public QGroupBox
{
    Q_OBJECT
public:
    explicit ClockDisplayBox(QWidget *parent = nullptr);

signals:

public slots:
    void updateFrequency(RfConfig::ClockType t, double f);

private:
    QHash<RfConfig::ClockType,QDoubleSpinBox*> d_boxes;

    // QWidget interface
public:
    QSize sizeHint() const override;
};

#endif // CLOCKDISPLAYBOX_H
