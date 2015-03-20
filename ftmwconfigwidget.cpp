#include "ftmwconfigwidget.h"
#include "ui_ftmwconfigwidget.h"

FtmwConfigWidget::FtmwConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FtmwConfigWidget)
{
    ui->setupUi(this);
}

FtmwConfigWidget::~FtmwConfigWidget()
{
    delete ui;
}
