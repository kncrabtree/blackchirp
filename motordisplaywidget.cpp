#include "motordisplaywidget.h"
#include "ui_motordisplaywidget.h"

MotorDisplayWidget::MotorDisplayWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MotorDisplayWidget)
{
    ui->setupUi(this);

    ui->zSlider1->setRange(0.0,1.0,2,2);
    ui->zSlider1->setAxis(MotorScan::MotorX);

    ui->zSlider2->setRange(0.0,1.0,2,2);
    ui->zSlider2->setAxis(MotorScan::MotorT);

    ui->xySlider1->setRange(0.0,1.0,2,2);
    ui->xySlider1->setAxis(MotorScan::MotorZ);

    ui->xySlider2->setRange(0.0,1.0,2,2);
    ui->xySlider2->setAxis(MotorScan::MotorT);

    ui->timeSlider1->setRange(0.0,1.0,2,2);
    ui->timeSlider1->setAxis(MotorScan::MotorX);

    ui->timeSlider2->setRange(0.0,1.0,2,2);
    ui->timeSlider2->setAxis(MotorScan::MotorY);

    ui->timeSlider3->setRange(0.0,1.0,2,2);
    ui->timeSlider3->setAxis(MotorScan::MotorZ);
}

MotorDisplayWidget::~MotorDisplayWidget()
{
    delete ui;
}
