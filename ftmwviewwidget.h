#ifndef FTMWVIEWWIDGET_H
#define FTMWVIEWWIDGET_H

#include <QWidget>

namespace Ui {
class FtmwViewWidget;
}

class FtmwViewWidget : public QWidget
{
    Q_OBJECT

public:
    explicit FtmwViewWidget(QWidget *parent = 0);
    ~FtmwViewWidget();

private:
    Ui::FtmwViewWidget *ui;
};

#endif // FTMWVIEWWIDGET_H
