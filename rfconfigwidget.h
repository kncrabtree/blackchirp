#ifndef RFCONFIGWIDGET_H
#define RFCONFIGWIDGET_H

#include <QWidget>

#include "rfconfig.h"
#include "clocktablemodel.h"

namespace Ui {
class RfConfigWidget;
}

class RfConfigWidget : public QWidget
{
    Q_OBJECT

public:
    explicit RfConfigWidget(QWidget *parent = 0);
    ~RfConfigWidget();

    void setRfConfig(const RfConfig c);
    RfConfig getRfConfig();

private:
    Ui::RfConfigWidget *ui;
    ClockTableModel *p_ctm;
};

#endif // RFCONFIGWIDGET_H
