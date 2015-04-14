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
    FtmwConfig getConfig() const;

    void lockFastFrame(const int nf);

public slots:
    void loadFromSettings();
    void saveToSettings();
    void configureUI();
    void validateSpinboxes();

private:
    Ui::FtmwConfigWidget *ui;

    void setComboBoxIndex(QComboBox *box, QVariant value);

};

#endif // FTMWCONFIGWIDGET_H
