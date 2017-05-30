#include "motorstatuswidget.h"
#include "ui_motorstatuswidget.h"

#include <QSettings>

#include "led.h"

MotorStatusWidget::MotorStatusWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MotorStatusWidget)
{
    ui->setupUi(this);

    d_x.label = new QLabel("X",this);
    d_x.negLimLed = new Led(this);
    d_x.positionBar = new QProgressBar(this);
    d_x.positionBar->setRange(0,10000);
    d_x.positionBar->setTextVisible(false);
    d_x.posLimLed = new Led(this);
    d_x.currentPos = 0.0;

    d_y.label = new QLabel("Y",this);
    d_y.negLimLed = new Led(this);
    d_y.positionBar = new QProgressBar(this);
    d_y.positionBar->setRange(0,10000);
    d_y.positionBar->setTextVisible(false);
    d_y.posLimLed = new Led(this);
    d_y.currentPos = 0.0;

    d_z.label = new QLabel("Z",this);
    d_z.negLimLed = new Led(this);
    d_z.positionBar = new QProgressBar(this);
    d_z.positionBar->setRange(0,10000);
    d_z.positionBar->setTextVisible(false);
    d_z.posLimLed = new Led(this);
    d_z.currentPos = 0.0;

    QList<AxisWidget> l;
    l << d_x << d_y << d_z;

    int row = 1;
    for(int i=0; i<l.size(); i++)
    {
        ui->gridLayout->addWidget(l.at(i).label,row,0);
        ui->gridLayout->addWidget(l.at(i).negLimLed,row,1);
        ui->gridLayout->addWidget(l.at(i).positionBar,row,2);
        ui->gridLayout->addWidget(l.at(i).posLimLed,row,3);
        row++;
    }

    ui->progressBar->setValue(0);

    updateRanges();

}

MotorStatusWidget::~MotorStatusWidget()
{
    delete ui;
}

void MotorStatusWidget::prepareForExperiment(const Experiment e)
{
    if(e.motorScan().isEnabled())
    {
        ui->progressBar->setRange(0,e.motorScan().xPoints()*e.motorScan().yPoints()*e.motorScan().zPoints()*e.motorScan().shotsPerPoint());
        ui->progressBar->setValue(0);
    }
    else
    {
        ui->progressBar->setRange(0,1);
        ui->progressBar->setValue(1);
    }
}

void MotorStatusWidget::updateRanges()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("motorController"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    s.beginReadArray(QString("channels"));

    s.setArrayIndex(0);
    d_x.minPos = s.value(QString("min"),-100.0).toDouble();
    d_x.maxPos = s.value(QString("max"),100.0).toDouble();
    s.setArrayIndex(1);
    d_y.minPos = s.value(QString("min"),-100.0).toDouble();
    d_y.maxPos = s.value(QString("max"),100.0).toDouble();
    s.setArrayIndex(2);
    d_z.minPos = s.value(QString("min"),-100.0).toDouble();
    d_z.maxPos = s.value(QString("max"),100.0).toDouble();

    s.endArray();
    s.endGroup();
    s.endGroup();

    updatePosition(BlackChirp::MotorX,d_x.currentPos);
    updatePosition(BlackChirp::MotorY,d_y.currentPos);
    updatePosition(BlackChirp::MotorZ,d_z.currentPos);
}

void MotorStatusWidget::updatePosition(BlackChirp::MotorAxis axis, double pos)
{
    AxisWidget *w;
    switch(axis)
    {
    case BlackChirp::MotorX:
        w = &d_x;
        break;
    case BlackChirp::MotorY:
        w = &d_y;
        break;
    case BlackChirp::MotorZ:
        w = &d_z;
        break;
    default:
        return;
        break;
    }

    w->currentPos = pos;
    int barPos = qBound(0,qRound((pos-w->minPos)/(w->maxPos-w->minPos)*10000.0),10000);
    w->positionBar->setValue(barPos);
}

void MotorStatusWidget::updateLimit(BlackChirp::MotorAxis axis, bool n, bool p)
{
    switch(axis)
    {
    case BlackChirp::MotorX:
        d_x.negLimLed->setState(n);
        d_x.posLimLed->setState(p);
        break;
    case BlackChirp::MotorY:
        d_y.negLimLed->setState(n);
        d_y.posLimLed->setState(p);
        break;
    case BlackChirp::MotorZ:
        d_z.negLimLed->setState(n);
        d_z.posLimLed->setState(p);
        break;
    default:
        break;
    }
}

void MotorStatusWidget::updateProgress(int s)
{
    ui->progressBar->setValue(s);
}
