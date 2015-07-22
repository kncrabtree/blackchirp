#ifndef EXPERIMENTVIEWWIDGET_H
#define EXPERIMENTVIEWWIDGET_H

#include <QWidget>

#include "experiment.h"

class QTabWidget;

class ExperimentViewWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ExperimentViewWidget(int num, QWidget *parent = 0);

signals:

public slots:

private:
    Experiment d_experiment;
    QTabWidget *p_tabWidget;
};

#endif // EXPERIMENTVIEWWIDGET_H
