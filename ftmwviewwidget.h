#ifndef FTMWVIEWWIDGET_H
#define FTMWVIEWWIDGET_H

#include <QWidget>
#include "fid.h"
#include <QList>

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
    void newFidList(QList<Fid> fl);
    void showFrame(int num);
    void fidTest(Fid f);


private:
    Ui::FtmwViewWidget *ui;

    QList<Fid> d_fidList;
};

#endif // FTMWVIEWWIDGET_H
