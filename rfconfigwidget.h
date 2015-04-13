#ifndef RFCONFIGWIDGET_H
#define RFCONFIGWIDGET_H

#include <QWidget>

namespace Ui {
class RfConfigWidget;
}

class RfConfigWidget : public QWidget
{
    Q_OBJECT

public:
    explicit RfConfigWidget(double tx, double rx, QWidget *parent = 0);
    ~RfConfigWidget();

signals:
    void setValonTx(const double);
    void setValonRx(const double);

public slots:
    void loadFromSettings();
    void txFreqUpdate(const double d);
    void rxFreqUpdate(const double d);
    void validate();
    void saveSettings();

private:
    Ui::RfConfigWidget *ui;

    double d_txSidebandSign, d_rxSidebandSign, d_valonTxFreq, d_valonRxFreq, d_currentRxMult;
    int sidebandIndex(const double d);
    double sideband(const int index);
};

#endif // RFCONFIGWIDGET_H
