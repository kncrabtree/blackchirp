#include "ftmwviewwidget.h"
#include "ui_ftmwviewwidget.h"

FtmwViewWidget::FtmwViewWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FtmwViewWidget)
{
    ui->setupUi(this);

    connect(ui->frameBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::showFrame);
}

FtmwViewWidget::~FtmwViewWidget()
{
    delete ui;
}

void FtmwViewWidget::newFidList(QList<Fid> fl)
{
    d_fidList = fl;
    if(ui->frameBox->maximum() != d_fidList.size())
    {
        if(ui->frameBox->value() > d_fidList.size())
        {
            blockSignals(true);
            ui->frameBox->setValue(d_fidList.size());
            blockSignals(false);
        }
        ui->frameBox->setMaximum(d_fidList.size());
    }
    showFrame(ui->frameBox->value());
}

void FtmwViewWidget::showFrame(int num)
{
    ui->ftPlot->newFid(d_fidList.at(num-1));
}

void FtmwViewWidget::fidTest(Fid f)
{
    ui->ftPlot->newFid(f);
}
