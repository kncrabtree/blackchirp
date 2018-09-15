#ifndef CLOCKDISPLAYWIDGET_H
#define CLOCKDISPLAYWIDGET_H

#include <QWidget>

#include "datastructs.h"

class QDoubleSpinBox;

class ClockDisplayWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ClockDisplayWidget(QWidget *parent = nullptr);

signals:

public slots:
    void updateFrequency(BlackChirp::ClockType t, double f);

private:
    QHash<BlackChirp::ClockType,QDoubleSpinBox*> d_boxes;
};

#endif // CLOCKDISPLAYWIDGET_H
