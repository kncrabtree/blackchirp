#ifndef DIGITIZERCONFIGWIDGET_H
#define DIGITIZERCONFIGWIDGET_H

#include <QWidget>

#include <src/data/experiment/ftmwconfig.h>


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
    void setComboBoxIndex(QComboBox *box, QVariant value);

private:
    Ui::DigitizerConfigWidget *ui;
    FtmwConfig d_config;

};

#endif // DIGITIZERCONFIGWIDGET_H
