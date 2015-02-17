#include "ftmwviewwidget.h"
#include "ui_ftmwviewwidget.h"

FtmwViewWidget::FtmwViewWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FtmwViewWidget)
{
    ui->setupUi(this);
}

FtmwViewWidget::~FtmwViewWidget()
{
    delete ui;
}
