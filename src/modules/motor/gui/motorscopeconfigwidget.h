#ifndef MOTORSCOPECONFIGWIDGET_H
#define MOTORSCOPECONFIGWIDGET_H

#include <QWidget>

#include <data/datastructs.h>

namespace Ui {
class MotorScopeConfigWidget;
}

class MotorScopeConfigWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MotorScopeConfigWidget(QWidget *parent = 0);
    ~MotorScopeConfigWidget();

    void setFromConfig(const BlackChirp::MotorScopeConfig &sc);
    BlackChirp::MotorScopeConfig toConfig() const;

private:
    Ui::MotorScopeConfigWidget *ui;
};

#endif // MOTORSCOPECONFIGWIDGET_H
