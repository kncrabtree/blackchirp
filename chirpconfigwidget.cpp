#include "chirpconfigwidget.h"
#include "ui_chirpconfigwidget.h"

ChirpConfigWidget::ChirpConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ChirpConfigWidget)
{
    ui->setupUi(this);
}

ChirpConfigWidget::~ChirpConfigWidget()
{
    delete ui;
}
