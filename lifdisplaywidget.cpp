#include "lifdisplaywidget.h"
#include "ui_lifdisplaywidget.h"

#include <QResizeEvent>

LifDisplayWidget::LifDisplayWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::LifDisplayWidget), d_numColumns(-1), d_numRows(-1)
{
    ui->setupUi(this);

    ui->freqSlicePlot->setXAxisTitle(QString::fromUtf16(u"Frequency (cm⁻¹)"));
    ui->freqSlicePlot->setTitle(QString("Frequency Slice"));
    ui->freqSlicePlot->setName(QString("freqSlicePlot"));

    ui->timeSlicePlot->setXAxisTitle(QString::fromUtf16(u"Time (µs)"));
    ui->timeSlicePlot->setTitle(QString("Time Slice"));
    ui->timeSlicePlot->setName(QString("timeSlicePlot"));

    ui->lifTracePlot->setTitle(QString("Current Trace"));

    connect(ui->lifTracePlot,&LifTracePlot::colorChanged,this,&LifDisplayWidget::lifColorChanged);

}

LifDisplayWidget::~LifDisplayWidget()
{
    delete ui;
}

void LifDisplayWidget::checkLifColors()
{
    ui->lifTracePlot->checkColors();
}

void LifDisplayWidget::resetLifPlot()
{
    ui->lifTracePlot->reset();
}

void LifDisplayWidget::lifShotAcquired(const LifTrace t)
{
    ui->lifTracePlot->newTrace(t);
}

void LifDisplayWidget::prepareForExperiment(const LifConfig c)
{
    ui->lifTracePlot->clearPlot();
    if(!c.isEnabled())
    {
        ui->timeSlicePlot->prepareForExperiment(0.0,1.0);
        ui->freqSlicePlot->prepareForExperiment(0.0,1.0);
    }
    else
    {
        QPair<double,double> delayRange = c.delayRange();
        ui->timeSlicePlot->prepareForExperiment(delayRange.first,delayRange.second);
        QPair<double,double> freqRange = c.frequencyRange();
        ui->freqSlicePlot->prepareForExperiment(freqRange.first,freqRange.second);
    }
}

void LifDisplayWidget::resizeEvent(QResizeEvent *ev)
{
    int margin = 5;
    ui->lifTracePlot->setGeometry(0,0,ev->size().width()/3-margin,2*ev->size().height()/5-margin);
    ui->timeSlicePlot->setGeometry(ev->size().width()/3,0,ev->size().width()/3-margin,2*ev->size().height()/5-margin);
    ui->freqSlicePlot->setGeometry(2*ev->size().width()/3,0,ev->size().width()/3-margin,2*ev->size().height()/5-margin);
    ui->lifSpectrogram->setGeometry(0,2*ev->size().height()/5,ev->size().width()-margin,3*ev->size().height()/5-margin);
}
