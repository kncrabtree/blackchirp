#ifndef LIFCONTROLWIDGET_H
#define LIFCONTROLWIDGET_H

#include <QWidget>

#include "datastructs.h"
#include "liftrace.h"
#include "lifconfig.h"

namespace Ui {
class LifControlWidget;
}

class LifControlWidget : public QWidget
{
    Q_OBJECT

public:
    explicit LifControlWidget(QWidget *parent = 0);
    ~LifControlWidget();

    LifConfig getSettings(LifConfig c);

signals:
    void updateScope(const BlackChirp::LifScopeConfig);
    void newTrace(const LifTrace);
    void lifColorChanged();

public slots:
    void scopeConfigChanged(const BlackChirp::LifScopeConfig c);
    void checkLifColors();

private:
    Ui::LifControlWidget *ui;

    BlackChirp::LifScopeConfig toConfig() const;
};

#endif // LIFCONTROLWIDGET_H
