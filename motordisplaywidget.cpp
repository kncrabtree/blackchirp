#include "motordisplaywidget.h"
#include "ui_motordisplaywidget.h"

MotorDisplayWidget::MotorDisplayWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MotorDisplayWidget)
{
    ui->setupUi(this);
}

MotorDisplayWidget::~MotorDisplayWidget()
{
    delete ui;
}
