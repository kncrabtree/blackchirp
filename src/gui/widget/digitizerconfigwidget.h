#ifndef DIGITIZERCONFIGWIDGET_H
#define DIGITIZERCONFIGWIDGET_H

#include <QWidget>

#include <data/experiment/ftmwconfig.h>


class QComboBox;

namespace Ui {
class DigitizerConfigWidget;
}

class DigitizerConfigWidget : public QWidget
{
    Q_OBJECT

public:
    explicit DigitizerConfigWidget(QWidget *parent = 0);
    ~DigitizerConfigWidget();

    void setFromConfig(const FtmwConfig config);
    FtmwConfig getConfig();

public slots:
    void configureUI();
    void validateSpinboxes();

private:
    Ui::DigitizerConfigWidget *ui;
    FtmwConfig d_config;

};

#endif // DIGITIZERCONFIGWIDGET_H
