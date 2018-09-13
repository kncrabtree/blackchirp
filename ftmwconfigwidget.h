#ifndef FTMWCONFIGWIDGET_H
#define FTMWCONFIGWIDGET_H

#include <QWidget>

#include "ftmwconfig.h"

class QComboBox;

namespace Ui {
class FtmwConfigWidget;
}

class FtmwConfigWidget : public QWidget
{
    Q_OBJECT

public:
    explicit FtmwConfigWidget(QWidget *parent = 0);
    ~FtmwConfigWidget();

    void setFromConfig(const FtmwConfig config);
    FtmwConfig getConfig();

public slots:
    void configureUI();
    void validateSpinboxes();

private:
    Ui::FtmwConfigWidget *ui;

    void setComboBoxIndex(QComboBox *box, QVariant value);
    FtmwConfig d_ftmwConfig;

};

#endif // FTMWCONFIGWIDGET_H
