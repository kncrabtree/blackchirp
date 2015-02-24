#ifndef FTMWVIEWWIDGET_H
#define FTMWVIEWWIDGET_H

#include <QWidget>
#include "fid.h"

namespace Ui {
class FtmwViewWidget;
}

class FtmwViewWidget : public QWidget
{
    Q_OBJECT

public:
    explicit FtmwViewWidget(QWidget *parent = 0);
    ~FtmwViewWidget();

public slots:
    void fidTest(Fid f);

private:
    Ui::FtmwViewWidget *ui;
};

#endif // FTMWVIEWWIDGET_H
