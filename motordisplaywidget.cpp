#include "motordisplaywidget.h"
#include "ui_motordisplaywidget.h"

MotorDisplayWidget::MotorDisplayWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MotorDisplayWidget)
{
    ui->setupUi(this);

    ui->spectrogramSlider1->setRange(0.0,1.0,2,2);
    ui->spectrogramSlider1->setAxis(MotorScan::MotorX);

    ui->spectrogramSlider2->setRange(0.0,1.0,2,2);
    ui->spectrogramSlider2->setAxis(MotorScan::MotorT);
}

MotorDisplayWidget::~MotorDisplayWidget()
{
    delete ui;
}
