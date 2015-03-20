#ifndef FTMWCONFIGWIDGET_H
#define FTMWCONFIGWIDGET_H

#include <QWidget>

namespace Ui {
class FtmwConfigWidget;
}

class FtmwConfigWidget : public QWidget
{
    Q_OBJECT

public:
    explicit FtmwConfigWidget(QWidget *parent = 0);
    ~FtmwConfigWidget();

private:
    Ui::FtmwConfigWidget *ui;
};

#endif // FTMWCONFIGWIDGET_H
